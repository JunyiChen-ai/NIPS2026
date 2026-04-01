"""
Faithful reproduction of 11 baseline methods using offline-extracted features.
Each method follows the original repo's implementation as closely as possible.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import faiss


# ============================================================
# 1. Geometry of Truth — LR Probe
# Original: geometry-of-truth/probes.py LRProbe
# Linear(d_in, 1, bias=False) + Sigmoid, trained with BCE, AdamW
# ============================================================
class LRProbe(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def pred(self, x):
        return self(x).round()

    @staticmethod
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        """Original: probes.py lines 17-28"""
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
        return probe


# ============================================================
# 2. Geometry of Truth — Mass-Mean Probe
# Original: geometry-of-truth/probes.py MMProbe
# Direction = pos_mean - neg_mean, with covariance-based correction
# ============================================================
class MMProbe(torch.nn.Module):
    def __init__(self, direction, inv):
        super().__init__()
        self.direction = torch.nn.Parameter(direction, requires_grad=False)
        self.inv = torch.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if iid:
            return torch.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return torch.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(acts, labels, atol=1e-3, device='cpu'):
        """Original: probes.py lines 56-67"""
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        inv = torch.linalg.pinv(covariance, hermitian=True, atol=atol)
        return MMProbe(direction, inv).to(device)


# ============================================================
# 3. No Answer Needed — PCA + Logistic Regression
# Original: correctness-model-internals/src/classifying/activations_handler.py
# PCA via SVD on centered data, then sklearn LogisticRegression
# ============================================================
def pca_lr_probe(train_acts, train_labels, test_acts, test_labels,
                 n_components=50):
    """Original: activations_handler.py lines 300-311, classification_utils.py"""
    # Center
    mean = train_acts.mean(dim=0)
    train_centered = train_acts - mean
    test_centered = test_acts - mean

    # PCA via SVD
    U, S, Vh = torch.linalg.svd(train_centered.float(), full_matrices=False)
    train_pca = (train_centered.float() @ Vh.T[:, :n_components]).numpy()
    test_pca = (test_centered.float() @ Vh.T[:, :n_components]).numpy()

    # StandardScaler + LogisticRegression
    scaler = StandardScaler()
    train_pca = scaler.fit_transform(train_pca)
    test_pca = scaler.transform(test_pca)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_pca, train_labels.numpy())
    probs = clf.predict_proba(test_pca)[:, 1]
    preds = clf.predict(test_pca)

    return probs, preds


# ============================================================
# 4. ITI — Direction per head
# Original: honest_llama/utils.py lines 262-375
# LogisticRegression per head, direction = probe coefficient
# ============================================================
def iti_directions(train_acts, train_labels, test_acts, test_labels):
    """
    train_acts: (N, n_layers, n_heads, head_dim)
    Returns per-head AUROC and accuracy.
    Original: utils.py line 345 — LogisticRegression(max_iter=1000)
    """
    n_layers, n_heads = train_acts.shape[1], train_acts.shape[2]
    results = np.zeros((n_layers, n_heads, 2))  # auroc, acc

    for li in range(n_layers):
        for hi in range(n_heads):
            X_train = train_acts[:, li, hi, :].numpy()
            y_train = train_labels.numpy()
            X_test = test_acts[:, li, hi, :].numpy()
            y_test = test_labels.numpy()

            clf = LogisticRegression(random_state=0, max_iter=1000)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            results[li, hi, 0] = roc_auc_score(y_test, probs)
            results[li, hi, 1] = accuracy_score(y_test, clf.predict(X_test))

    return results


# ============================================================
# 5. Knowledge Boundary — MLP
# Original: LLM-Knowledge-Boundary-Perception-via-Internal-States/
#           hidden_state_detection/models.py lines 5-22
# 4-layer MLP: Linear(d→512)→ReLU→Linear(512→64)→ReLU→
#              Linear(64→32)→ReLU→Dropout(0.5)→Linear(32→2)→Softmax
# ============================================================
class KBNet(torch.nn.Module):
    def __init__(self, d_in, dropout=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x):
        return torch.nn.functional.softmax(self.net(x), dim=-1)

    @staticmethod
    def train_and_eval(train_acts, train_labels, test_acts, test_labels,
                       lr=5e-5, epochs=100, batch_size=64, device='cpu'):
        """Original: engine.py lines 36-135"""
        d_in = train_acts.shape[-1]
        model = KBNet(d_in).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_acts = train_acts.to(device)
        train_labels = train_labels.long().to(device)

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_acts))
            for i in range(0, len(train_acts), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                out = model(train_acts[idx])
                loss = criterion(out, train_labels[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            probs = model(test_acts.to(device))[:, 1].cpu().numpy()
            preds = probs > 0.5
        return probs, preds


# ============================================================
# 6. LID — Local Intrinsic Dimension
# Original: lid-hallucinationdetection/src/lids.py lines 14-41
# KNN-based LID with FAISS, k=200
# ============================================================
def compute_lid(train_acts, test_acts, k=200):
    """
    Original: lids.py lines 25-40
    train_acts: (N_train, dim) — reference set (correct predictions)
    test_acts: (N_test, dim) — samples to score
    Returns: LID scores (N_test,)
    """
    train_np = np.ascontiguousarray(train_acts.float().numpy())
    test_np = np.ascontiguousarray(test_acts.float().numpy())
    d = train_np.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(train_np)

    actual_k = min(k, len(train_np) - 1)
    distances, _ = index.search(test_np, actual_k)
    distances = np.sqrt(distances)  # FAISS returns squared L2

    # LID computation: -1 / mean(log(dist / max_dist))
    rk = distances[:, -1:]  # max distance (k-th neighbor)
    rk = np.maximum(rk, 1e-10)
    ratios = distances / rk
    ratios = np.maximum(ratios, 1e-10)
    lids = -1.0 / np.mean(np.log(ratios), axis=1)

    return lids


# ============================================================
# 7. Attention Satisfies — LR on attn_value_norms
# Original: mechanistic-error-probe/main_probe.py lines 64-79
# StandardScaler + LogisticRegression(L1, C=0.05)
# ============================================================
def attention_satisfies_probe(train_norms, train_labels, test_norms, test_labels):
    """
    train_norms: (N, n_layers, n_heads, prompt_len) — flatten to feature vector
    Original: main_probe.py line 79 — LogReg(penalty='l1', solver='liblinear', C=0.05)
    """
    # Flatten: use mean over prompt positions per head per layer
    train_feat = train_norms.float().mean(dim=-1).reshape(len(train_norms), -1).numpy()
    test_feat = test_norms.float().mean(dim=-1).reshape(len(test_norms), -1).numpy()

    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    test_feat = scaler.transform(test_feat)

    clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, max_iter=1000)
    clf.fit(train_feat, train_labels.numpy())
    probs = clf.predict_proba(test_feat)[:, 1]
    preds = clf.predict(test_feat)

    return probs, preds


# ============================================================
# 8. LLM-Check — Attention diagonal scoring
# Original: common_utils.py lines 319-352
# Score = sum over heads of mean(log(diagonal(attn_matrix)))
# We use pre-computed diag_logmean from attn_stats[:, :, 2]
# ============================================================
def llm_check_score(attn_stats):
    """
    attn_stats: (N, n_layers, n_heads, 3) — index 2 is diag_logmean
    Original: common_utils.py line 350 — torch.log(torch.diagonal(Sigma, 0)).mean()
    Returns: (N,) score per sample
    """
    # Sum diag_logmean across all heads, use a single layer (original uses layer_num=15)
    # For our 28-layer model, use layer 14 (0-indexed middle-ish)
    diag_logmean = attn_stats[:, :, :, 2]  # (N, n_layers, n_heads)
    # Original sums over heads for a single layer
    scores = diag_logmean[:, 14, :].sum(dim=-1)  # (N,)
    return scores.numpy()


# ============================================================
# 9. SEP — Logistic Regression on generation hidden states
# Original: semantic-entropy-probes train-latent-probe.ipynb
# LogisticRegression(default) on per-layer hidden states
# ============================================================
def sep_probe(train_acts, train_labels, test_acts, test_labels, layer=-1):
    """
    train_acts: (N, n_layers+2, hidden_dim)
    layer: which layer to use (-1 = last = norm output)
    Original: LogisticRegression().fit(X_train, y_train)
    """
    X_train = train_acts[:, layer, :].float().numpy()
    X_test = test_acts[:, layer, :].float().numpy()
    y_train = train_labels.numpy()
    y_test = test_labels.numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    return probs, preds


# ============================================================
# 10. CoE — Chain of Embedding scores (unsupervised)
# Original: Chain-of-Embedding/score.py lines 48-104
# 4 geometric scores from layer-wise hidden state changes
# ============================================================
def compute_coe_scores(hidden_states):
    """
    hidden_states: (N, n_layers+2, hidden_dim) — mean-pooled across tokens
    Returns: dict of 4 scores, each (N,)
    Original: score.py lines 52-104
    """
    N = hidden_states.shape[0]
    hs = hidden_states.float()  # (N, L, D)

    # CoE-Mag: normalized layer-wise L2 distance
    # norm_denom = ||hs[-1] - hs[0]||
    norm_denom = torch.norm(hs[:, -1, :] - hs[:, 0, :], dim=-1, keepdim=True)  # (N, 1)
    norm_denom = torch.clamp(norm_denom, min=1e-10)

    n_layers_plus = hs.shape[1]
    diffs = []
    for i in range(n_layers_plus - 1):
        diff_norm = torch.norm(hs[:, i + 1, :] - hs[:, i, :], dim=-1) / norm_denom.squeeze(-1)
        diffs.append(diff_norm)
    diffs = torch.stack(diffs, dim=1)  # (N, L-1)
    coe_mag = diffs.mean(dim=1)  # (N,)

    # CoE-Ang: normalized angular change
    angles = []
    for i in range(n_layers_plus - 1):
        cos_sim = torch.nn.functional.cosine_similarity(hs[:, i + 1, :], hs[:, i, :], dim=-1)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        ang = torch.arccos(cos_sim)
        angles.append(ang)
    angles = torch.stack(angles, dim=1)  # (N, L-1)

    # Baseline angle
    cos_baseline = torch.nn.functional.cosine_similarity(hs[:, -1, :], hs[:, 0, :], dim=-1)
    cos_baseline = torch.clamp(cos_baseline, -1 + 1e-7, 1 - 1e-7)
    baseline_ang = torch.arccos(cos_baseline)  # (N,)
    baseline_ang = torch.clamp(baseline_ang, min=1e-10)

    norm_angles = angles / baseline_ang.unsqueeze(-1)
    coe_ang = norm_angles.mean(dim=1)  # (N,)

    # CoE-R and CoE-C
    coe_r = coe_mag - coe_ang
    coe_c = torch.sqrt(coe_mag ** 2 + coe_ang ** 2 - 2 * coe_mag * coe_ang * torch.cos(
        torch.arctan(coe_ang / torch.clamp(coe_mag, min=1e-10))))

    return {
        "coe_mag": coe_mag.numpy(),
        "coe_ang": coe_ang.numpy(),
        "coe_r": coe_r.numpy(),
        "coe_c": coe_c.numpy(),
    }


# ============================================================
# 11. SeaKR — Energy score from logits
# Original: SeaKR/vllm_uncertainty/vllm/model_executor/layers/sampler.py line 67
# energy_score = logsumexp(logits)
# ============================================================
def seakr_energy_score(logit_stats):
    """
    logit_stats: list of dicts, each with 'logsumexp' key
    Original: sampler.py line 67 — torch.logsumexp(logits, dim=-1)
    Returns: (N,) energy scores
    """
    return np.array([s.get("logsumexp", 0.0) for s in logit_stats])


# ============================================================
# 12. STEP — MLP Scorer
# Original: STEP/STEP/train_scorer/train_scorer.py lines 83-87
# 2-layer MLP: Linear(d→512)→ReLU→Linear(512→1)
# BCEWithLogitsLoss, Adam lr=1e-4, early stopping patience=5
# ============================================================
class STEPScorer(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @staticmethod
    def train_and_eval(train_acts, train_labels, test_acts, test_labels,
                       lr=1e-4, weight_decay=1e-5, epochs=100,
                       batch_size=128, patience=5, device='cpu'):
        """Original: train_scorer.py lines 270-360"""
        d_in = train_acts.shape[-1]
        model = STEPScorer(d_in).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # pos_weight for class imbalance
        n_pos = train_labels.sum().item()
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_acts = train_acts.float().to(device)
        train_labels = train_labels.float().to(device)

        best_loss = float('inf')
        patience_counter = 0

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_acts))
            epoch_loss = 0.0
            for i in range(0, len(train_acts), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                out = model(train_acts[idx])
                loss = criterion(out, train_labels[idx])
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.eval()
        with torch.no_grad():
            logits = model(test_acts.float().to(device)).cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)

        return probs, preds
