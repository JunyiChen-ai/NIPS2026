"""GPU classifier wrappers — drop-in replacements for sklearn LR / GBT.

Used in the new setting (--setting new) where the inner cross-val sklearn calls
become the runtime bottleneck. Old setting keeps using sklearn directly.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch
import torch.nn.functional as F


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_t(X):
    if isinstance(X, torch.Tensor):
        return X.to(dtype=torch.float32, device=_DEVICE)
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=_DEVICE)


def _sklearn_lr_fallback(X_tr, y_tr, Xs, C, max_iter):
    """Fallback path: sklearn LR on CPU. Returns list of positive-class probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xt = sc.fit_transform(np.asarray(X_tr))
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
    clf.fit(Xt, np.asarray(y_tr))
    return [clf.predict_proba(sc.transform(np.asarray(X)))[:, 1] for X in Xs]


def gpu_lr_fit_predict(X_tr, y_tr, *Xs, C=1.0, max_iter=200, standardize=True):
    """Logistic regression on GPU via LBFGS. Drop-in for sklearn LR's
    .fit() + .predict_proba(...)[:, 1].

    Returns a list of np.ndarray (one per X in Xs), each shape (n,) — positive-class probabilities.

    On CUDA OOM (e.g. when another process is hogging the GPU), transparently
    falls back to sklearn LR on CPU so the pipeline keeps making progress.
    """
    try:
        Xt = _to_t(X_tr)
        yt = torch.as_tensor(np.asarray(y_tr), dtype=torch.float32, device=_DEVICE)
        if standardize:
            mu = Xt.mean(0)
            sd = Xt.std(0).clamp_min(1e-6)
            Xt = (Xt - mu) / sd
        D = Xt.shape[1]
        n = Xt.shape[0]
        W = torch.zeros(D, device=_DEVICE, requires_grad=True)
        b = torch.zeros(1, device=_DEVICE, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter,
                                 tolerance_grad=1e-7, history_size=20)
        l2 = 1.0 / max(C, 1e-8)
        def closure():
            opt.zero_grad()
            logits = Xt @ W + b
            loss = F.binary_cross_entropy_with_logits(logits, yt) + l2 * (W * W).sum() / max(n, 1)
            loss.backward()
            return loss
        opt.step(closure)
        out = []
        with torch.no_grad():
            for X in Xs:
                Xv = _to_t(X)
                if standardize:
                    Xv = (Xv - mu) / sd
                out.append(torch.sigmoid(Xv @ W + b).cpu().numpy())
        del Xt, yt, W, b
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return out
    except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
        # OOM or other CUDA error → drop to CPU sklearn path
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            if _DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            return _sklearn_lr_fallback(X_tr, y_tr, Xs, C, max_iter)
        raise


class XGBHGBShim:
    """Drop-in shim that mimics sklearn HistGradientBoostingClassifier's
    interface using xgboost-gpu under the hood. Used for monkey-patching.

    Maps sklearn args → xgboost args:
        max_leaf_nodes  → max_leaves
        max_iter        → n_estimators
        min_samples_leaf → min_child_weight
        l2_regularization → reg_lambda
        learning_rate   → learning_rate
    """
    def __init__(self, max_leaf_nodes=31, learning_rate=0.1, max_iter=100,
                 min_samples_leaf=20, l2_regularization=0.0, random_state=None,
                 **kwargs):
        self._params = dict(
            max_leaves=max_leaf_nodes,
            learning_rate=learning_rate,
            n_estimators=max_iter,
            min_child_weight=min_samples_leaf,
            reg_lambda=l2_regularization,
            random_state=random_state,
            verbosity=0,
        )
        self._clf = None
        self._fallback = False

    def _build_gpu(self):
        import xgboost as xgb
        return xgb.XGBClassifier(
            tree_method="hist",
            device="cuda" if _DEVICE.type == "cuda" else "cpu",
            **self._params,
        )

    def _build_cpu(self):
        from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
        return _HGB(
            max_leaf_nodes=self._params["max_leaves"],
            learning_rate=self._params["learning_rate"],
            max_iter=self._params["n_estimators"],
            min_samples_leaf=self._params["min_child_weight"],
            l2_regularization=self._params["reg_lambda"],
            random_state=self._params["random_state"],
        )

    def fit(self, X, y, **kw):
        Xa, ya = np.asarray(X), np.asarray(y)
        try:
            self._clf = self._build_gpu()
            self._clf.fit(Xa, ya)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg or "gpu" in msg:
                # GPU OOM/error → fall back to sklearn HGB on CPU
                self._fallback = True
                self._clf = self._build_cpu()
                self._clf.fit(Xa, ya)
            else:
                raise
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(np.asarray(X))

    def predict(self, X):
        return self._clf.predict(np.asarray(X))


def get_hgb_class(use_gpu: bool):
    """Return HistGradientBoostingClassifier-compatible class.
    If use_gpu and xgboost is available, returns the GPU shim; otherwise
    returns the original sklearn class. Drop-in for `clf = HGB(...)` calls.
    """
    if use_gpu:
        try:
            import xgboost  # noqa: F401
            return XGBHGBShim
        except ImportError:
            warnings.warn("xgboost not installed; falling back to sklearn HGB.")
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
    return _HGB


def gpu_gbt_fit_predict(X_tr, y_tr, *Xs, n_estimators=200, max_leaves=8,
                        learning_rate=0.1, min_child_weight=10, reg_lambda=0.5,
                        random_state=42):
    """xgboost-gpu drop-in for sklearn HistGradientBoostingClassifier.

    Returns list of np.ndarray (positive-class probabilities) per X in Xs.
    Falls back to sklearn HGB if xgboost unavailable.
    """
    try:
        import xgboost as xgb
    except ImportError:
        warnings.warn("xgboost not installed; falling back to sklearn HGB on CPU.")
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaves, learning_rate=learning_rate,
            max_iter=n_estimators, min_samples_leaf=min_child_weight,
            l2_regularization=reg_lambda, random_state=random_state,
        )
        clf.fit(np.asarray(X_tr), np.asarray(y_tr))
        return [clf.predict_proba(np.asarray(X))[:, 1] for X in Xs]

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_leaves=max_leaves,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        tree_method="hist",
        device="cuda" if _DEVICE.type == "cuda" else "cpu",
        random_state=random_state,
        verbosity=0,
    )
    clf.fit(np.asarray(X_tr), np.asarray(y_tr))
    return [clf.predict_proba(np.asarray(X))[:, 1] for X in Xs]
