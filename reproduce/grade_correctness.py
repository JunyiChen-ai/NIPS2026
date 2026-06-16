"""
Per-(model, dataset) correctness grader.

Reads meta.json (texts, gen_texts, gold_answers) and produces a 0/1 label per sample
indicating whether the model's generation matched the gold answer. Output:
  reproduce/correctness_labels/{model}/{dataset}/labels.json

Each dataset has its own grader because answer formats differ:
  gsm8k        — gold is integer; pull last "Final answer: N" / "Answer: N" / \\boxed{N}
  math         — gold is short string (number, letter, or symbolic); pull \\boxed{...}
  theoremqa    — gold is float/int/list; pull "Therefore, the answer is X" / final number
  mmlu         — gold is letter A-D; pull "Answer: X" / \\boxed{X}
  commonsenseqa— gold is letter A-E; same as mmlu
  belebele     — gold is digit 1-4 (1-indexed); pull letter then map to digit

Phase-2 classification datasets (gold_answer is a fixed token):
  fava                — gold ∈ {"yes","no"}
  ragtruth            — gold ∈ {"yes","no"}
  common_claim_3class — gold ∈ {"True","False","Neither"}
  when2call_3class    — gold ∈ {"A","B","C"}
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Strip markdown bold/italic so "**Final answer: 21**" parses
_MARKDOWN_RE = re.compile(r"(\*\*|__|\*|`|_)")
def _strip_markdown(s: str) -> str:
    return _MARKDOWN_RE.sub("", s)

# Normalize numbers: drop commas + currency + trailing punctuation
def _normalize_num_str(s: str) -> str:
    s = s.strip().rstrip(".,;:!?)")
    s = s.lstrip("$€£¥")
    s = s.replace(",", "")
    return s

def _to_float(s: str):
    s = _normalize_num_str(s)
    # Handle fractions a/b
    if "/" in s and s.count("/") == 1:
        a, b = s.split("/")
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return None

def _num_close(pred: str, gold: str, tol: float = 1e-3, rel_tol: float = 1e-3) -> bool:
    p = _to_float(pred)
    g = _to_float(gold)
    if p is None or g is None:
        return _normalize_num_str(pred) == _normalize_num_str(gold)
    if abs(p - g) <= tol:
        return True
    if abs(g) > 1e-9 and abs(p - g) / abs(g) <= rel_tol:
        return True
    return False

# ---------------------------------------------------------------------------
# Number extraction (for gsm8k / math / theoremqa)
# ---------------------------------------------------------------------------

# Strict patterns in priority order. Last match within each pattern wins
# (model often re-states answer at the end).
# Each pattern captures exactly one group: the answer string.
_NUM_PATTERNS = [
    # \boxed{...}
    re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}"),
    # "Final answer: X" / "final answer is X"
    re.compile(r"(?:[Ff]inal\s+answer|FINAL ANSWER)\s*(?:is|=|:|：)?\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"),
    # "Answer: X" (must come after Final-answer pattern so "Final answer" wins)
    re.compile(r"(?<![Ff]inal )(?<![Ff]inal_)[Aa]nswer\s*(?:is|=|:|：)\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"),
    # "Therefore, the answer is X" (theoremqa style)
    re.compile(r"(?:[Tt]herefore|[Tt]hus|[Hh]ence)[^.]{0,80}?(?:answer|result|value)\s*(?:is|=|:|：)?\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"),
    # "= X" at end of last math line
    re.compile(r"=\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\.?\s*$", re.MULTILINE),
]

def extract_number(gen: str) -> str | None:
    """Try patterns in priority order, return last match of highest-priority pattern hit."""
    g = _strip_markdown(gen)
    for pat in _NUM_PATTERNS:
        matches = pat.findall(g)
        if matches:
            return matches[-1]
    # Fallback: last bare number in text
    bare = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", g)
    if bare:
        return bare[-1]
    return None


# ---------------------------------------------------------------------------
# Letter extraction (for mmlu / commonsenseqa / belebele)
# ---------------------------------------------------------------------------

# Letter must be immediately followed by ) or word boundary + non-letter, to avoid
# matching "As we discussed". We also accept \boxed{X} and Answer: X.
_LETTER_PATTERNS = [
    re.compile(r"\\boxed\s*\{\s*\(?([A-Z])\)?\s*\}"),
    re.compile(r"(?:[Ff]inal\s+answer|FINAL ANSWER)\s*(?:is|=|:|：)?\s*\(?([A-Z])\)(?![A-Za-z])"),
    re.compile(r"(?:[Ff]inal\s+answer|FINAL ANSWER)\s*(?:is|=|:|：)?\s*([A-Z])(?![A-Za-z])"),
    re.compile(r"(?<![Ff]inal )[Aa]nswer\s*(?:is|=|:|：)\s*\(?([A-Z])\)(?![A-Za-z])"),
    re.compile(r"(?<![Ff]inal )[Aa]nswer\s*(?:is|=|:|：)\s*([A-Z])(?![A-Za-z])"),
    # "the answer is (X)"
    re.compile(r"the\s+(?:correct\s+)?(?:answer|choice|option)\s+is\s*\(?([A-Z])\)(?![A-Za-z])"),
    # bare "(X)" near end (last occurrence)
    re.compile(r"\(([A-Z])\)(?=[\s.,!?]|$)"),
]

def extract_letter(gen: str, valid_letters: str = "ABCDE") -> str | None:
    g = _strip_markdown(gen)
    for pat in _LETTER_PATTERNS:
        matches = pat.findall(g)
        # Filter by valid range
        valid_matches = [m for m in matches if m in valid_letters]
        if valid_matches:
            return valid_matches[-1]
    return None


# ---------------------------------------------------------------------------
# Per-dataset graders
# ---------------------------------------------------------------------------

def grade_gsm8k(gen: str, gold: str) -> int:
    pred = extract_number(gen)
    if pred is None:
        return 0
    return 1 if _num_close(pred, gold) else 0

def grade_math(gen: str, gold: str) -> int:
    """MATH gold is short string: number, letter, or symbolic (e.g. \\frac{3}{4}).
    Order: letter > numeric > symbolic. Symbolic uses normalized boxed content."""
    g = _strip_markdown(gen)
    gold_clean = gold.strip()

    # Case: gold is a single letter
    if len(gold_clean) == 1 and gold_clean.isalpha() and gold_clean.isupper():
        pred = extract_letter(g, valid_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return 1 if pred == gold_clean else 0

    # Symbolic-style gold (contains LaTeX commands like \frac, \sqrt, etc.)
    is_symbolic = ("\\" in gold_clean) or ("{" in gold_clean) or ("^" in gold_clean)

    # Always try \boxed{} first for both numeric and symbolic golds
    # (boxed content is the model's *canonical* answer)
    # Use balanced brace matching for nested LaTeX like \boxed{\frac{3}{4}}
    def _find_boxed(text):
        results = []
        i = 0
        while True:
            idx = text.find("\\boxed{", i)
            if idx == -1:
                break
            depth = 1
            j = idx + len("\\boxed{")
            start = j
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                results.append(text[start:j-1])
            i = j
        return results

    boxed_list = _find_boxed(g)
    norm = lambda x: re.sub(r"[\s$().,]", "", x).replace("\\\\", "\\").lower()

    if is_symbolic:
        if boxed_list:
            pred = boxed_list[-1].strip()
            return 1 if norm(pred) == norm(gold_clean) else 0
        return 0

    # Numeric gold path
    if _to_float(gold_clean) is not None:
        # First check boxed
        if boxed_list:
            pred = boxed_list[-1].strip()
            if _to_float(pred) is not None and _num_close(pred, gold_clean):
                return 1
            # If boxed is non-numeric (e.g. "\frac{1}{2}" with numeric gold "0.5"),
            # try interpreting fraction
            if "/" in pred and "\\frac" not in pred and _num_close(pred, gold_clean):
                return 1
        # Fall back to number extraction
        pred = extract_number(g)
        if pred is None:
            return 0
        return 1 if _num_close(pred, gold_clean) else 0

    # Pure string fallback
    if boxed_list:
        pred = boxed_list[-1].strip()
        return 1 if norm(pred) == norm(gold_clean) else 0
    return 0

def grade_theoremqa(gen: str, gold: str) -> int:
    """TheoremQA gold may be: float, int, '[1, 2, 3]' list, or 'True'/'False'."""
    gold_clean = gold.strip()
    g = _strip_markdown(gen)

    # Bool case
    if gold_clean.lower() in ("true", "false"):
        # Look for "True"/"False" near "answer" / "Therefore"
        for pat in [
            r"(?:[Tt]herefore|[Tt]hus|answer\s+is)[^.]{0,60}?(True|False|true|false|TRUE|FALSE)",
            r"\\boxed\s*\{\s*(True|False|true|false)\s*\}",
        ]:
            m = re.findall(pat, g)
            if m:
                return 1 if m[-1].lower() == gold_clean.lower() else 0
        return 0

    # List case
    if gold_clean.startswith("["):
        # Find any list literal in gen
        lists = re.findall(r"\[\s*[\d\s,.\-+]+\s*\]", g)
        for L in lists[::-1]:
            try:
                pred_list = [float(x.strip()) for x in L.strip("[]").split(",") if x.strip()]
                gold_list = [float(x.strip()) for x in gold_clean.strip("[]").split(",") if x.strip()]
                if len(pred_list) == len(gold_list) and all(
                    abs(p - go) < 1e-3 for p, go in zip(pred_list, gold_list)
                ):
                    return 1
            except ValueError:
                continue
        return 0

    # Numeric case (most common)
    pred = extract_number(g)
    if pred is None:
        return 0
    return 1 if _num_close(pred, gold_clean) else 0


def grade_mmlu(gen: str, gold: str) -> int:
    pred = extract_letter(gen, valid_letters="ABCD")
    return 1 if pred == gold.strip().upper() else 0

def grade_commonsenseqa(gen: str, gold: str) -> int:
    pred = extract_letter(gen, valid_letters="ABCDE")
    return 1 if pred == gold.strip().upper() else 0

def grade_belebele(gen: str, gold: str) -> int:
    """Gold is '1'-'4' (1-indexed). Map letter A-D -> 1-4."""
    pred = extract_letter(gen, valid_letters="ABCD")
    if pred is None:
        return 0
    pred_digit = str(ord(pred) - ord("A") + 1)
    return 1 if pred_digit == gold.strip() else 0


# ---------------------------------------------------------------------------
# Phase 2 classification graders (gold is one of a small fixed set of tokens).
# All 4 share the same shape: pull whatever follows "Final answer:" anywhere in
# the gen, normalize, compare token-equality with gold (case-insensitive for
# yes/no/True/False/Neither, case-preserving for A/B/C).
# ---------------------------------------------------------------------------

# Strip outer markdown bold/italic + (), period/comma, surrounding quotes.
def _strip_token(s: str) -> str:
    s = s.strip()
    s = s.strip("()*_`'\"")
    s = s.rstrip(".,;:!?")
    return s

# Capture the FIRST whitespace-delimited token after "Final answer:" anywhere in
# gen (Mistral often puts it at the start; Qwen/Llama at the end). Both forms
# parse correctly because the regex is global, not anchored.
_FINAL_ANSWER_RE = re.compile(
    r"[Ff]inal\s+[Aa]nswer\s*(?:is|=|:|：)?\s*\(?\s*\*{0,2}\s*([A-Za-z]+)",
    re.IGNORECASE,
)

def _extract_final_answer_token(gen: str) -> str | None:
    """Return the first token after 'Final answer:' or None."""
    m = _FINAL_ANSWER_RE.search(gen)
    if not m:
        return None
    return _strip_token(m.group(1))


def grade_fava(gen: str, gold: str) -> int:
    """Gold ∈ {'yes','no'}. Case-insensitive token match."""
    pred = _extract_final_answer_token(gen)
    if pred is None:
        return 0
    return 1 if pred.lower() == gold.strip().lower() else 0


def grade_ragtruth(gen: str, gold: str) -> int:
    return grade_fava(gen, gold)  # same alphabet


def grade_common_claim_3class(gen: str, gold: str) -> int:
    """Gold ∈ {'True','False','Neither'}. Case-insensitive match."""
    pred = _extract_final_answer_token(gen)
    if pred is None:
        return 0
    return 1 if pred.lower() == gold.strip().lower() else 0


def grade_when2call_3class(gen: str, gold: str) -> int:
    """Gold ∈ {'A','B','C'}. Case-sensitive single letter."""
    pred = _extract_final_answer_token(gen)
    if pred is None:
        return 0
    # Letter must be single uppercase A/B/C; tolerate parens or lowercase
    p = pred.strip().upper()
    if len(p) != 1 or p not in "ABC":
        return 0
    return 1 if p == gold.strip().upper() else 0


GRADERS = {
    "gsm8k": grade_gsm8k,
    "math": grade_math,
    "theoremqa": grade_theoremqa,
    "mmlu": grade_mmlu,
    "commonsenseqa": grade_commonsenseqa,
    "belebele": grade_belebele,
    "fava": grade_fava,
    "ragtruth": grade_ragtruth,
    "common_claim_3class": grade_common_claim_3class,
    "when2call_3class": grade_when2call_3class,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def grade_meta(meta: dict, dataset: str) -> list[int]:
    grader = GRADERS[dataset]
    return [grader(g, str(go)) for g, go in zip(meta["gen_texts"], meta["gold_answers"])]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-root", default="/home/junyi/b2-nips/extraction/features",
                    help="Root containing {model}/{dataset}/all/meta.json")
    ap.add_argument("--out-root", default="/home/junyi/NIPS2026/reproduce/correctness_labels")
    ap.add_argument("--models", nargs="+",
                    default=["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"])
    ap.add_argument("--datasets", nargs="+",
                    default=["gsm8k", "math", "mmlu", "commonsenseqa", "belebele", "theoremqa",
                             "fava", "ragtruth", "common_claim_3class", "when2call_3class"])
    args = ap.parse_args()
    META_ROOT = Path(args.meta_root)
    OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = []
    for m in args.models:
        for d in args.datasets:
            # B2 layout: features/{model}/{dataset}/all/meta.json
            meta_path = META_ROOT / m / d / "all" / "meta.json"
            if not meta_path.exists():
                # Fallback: features/{model}/{dataset}/meta.json (legacy)
                meta_path_alt = META_ROOT / m / d / "meta.json"
                if meta_path_alt.exists():
                    meta_path = meta_path_alt
                else:
                    print(f"  skip {m}/{d}: meta.json not found at {meta_path}")
                    continue
            with open(meta_path) as f:
                meta = json.load(f)
            if "gold_answers" not in meta:
                print(f"  skip {m}/{d}: meta.json missing 'gold_answers' field")
                continue
            labels = grade_meta(meta, d)
            N = len(labels)
            n_pos = sum(labels)
            out_dir = OUT_ROOT / m / d
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "labels.json", "w") as f:
                json.dump({"labels": labels, "n_pos": n_pos, "n_total": N,
                           "pos_rate": n_pos / N}, f)
            summary.append((m, d, N, n_pos, n_pos / N))
            print(f"  {m:18s} {d:18s} N={N:5d}  pos={n_pos:5d}  rate={100*n_pos/N:5.1f}%")
    return summary


if __name__ == "__main__":
    main()
