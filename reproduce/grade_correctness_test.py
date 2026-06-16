"""
Spot-check the grader on tricky cases. Each test asserts (gen, gold) -> expected label.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from grade_correctness import (
    grade_gsm8k, grade_math, grade_theoremqa,
    grade_mmlu, grade_commonsenseqa, grade_belebele,
    grade_fava, grade_ragtruth, grade_common_claim_3class, grade_when2call_3class,
    extract_number, extract_letter,
)

# ---------- gsm8k ----------
G = [
    # bold-wrapped
    ("**Final answer: 21**", "21", 1),
    # currency $
    ("Final answer: $18", "18", 1),
    # comma thousand
    ("Final answer: $202,650", "202650", 1),
    # plain "Answer: 18"
    ("So she makes 9*2=18 dollars. Answer: 18", "18", 1),
    # \boxed
    ("Total = 3 bolts.\n\\[\n\\boxed{3}\n\\]", "3", 1),
    # wrong
    ("Final answer: 53.2", "460", 0),
    # truncated mid-number — model said "Final answer: 5" when it meant 50
    # Per current setup we trust last extract; wrong but conservative
    ("Final distance from home = 50 miles.\n\nFinal answer: 5", "45", 0),
    # no answer found
    ("Step 1: ... Step 2: ...", "100", 0),
    # final answer with period
    ("Final answer: 9.\n", "9", 1),
    # decimal close to gold
    ("Final answer: 18.0", "18", 1),
    # fraction
    ("Final answer: 1/2", "0.5", 1),
]
for gen, gold, exp in G:
    got = grade_gsm8k(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [gsm8k]  {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:80s}")

# ---------- math ----------
M = [
    # boxed letter
    ("the answer is \\boxed{H}.", "H", 1),
    ("\\boxed{A}", "A", 1),
    # boxed letter wrong
    ("\\boxed{D}", "B", 0),
    # numeric gold, boxed
    ("\\boxed{60}", "60", 1),
    # numeric gold, "Final answer: X"
    ("Final answer: 60", "60", 1),
    # symbolic gold
    ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}", 1),
]
for gen, gold, exp in M:
    got = grade_math(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [math]   {status}  gold={gold!r:18s} pred_label={got} exp={exp}  gen={gen!r:60s}")

# ---------- theoremqa ----------
T = [
    # numeric
    ("Therefore, the answer is 0.02.", "0.02", 1),
    # bool
    ("Hence, the answer is True.", "True", 1),
    # list
    ("Therefore, the answer is [1, 2, 3].", "[1, 2, 3]", 1),
    # large number with rel tol
    ("the answer is 924.5", "924.0", 1),  # rel tol 1e-3 → tolerance ~1, pass
    # wrong
    ("Therefore, the answer is 0.12", "0.02", 0),
]
for gen, gold, exp in T:
    got = grade_theoremqa(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [theorem]{status}  gold={gold!r:18s} pred_label={got} exp={exp}  gen={gen!r:60s}")

# ---------- mmlu ----------
MM = [
    ("Answer: (C) 2,3", "C", 1),
    ("Final answer: D", "D", 1),
    ("\\boxed{D}", "D", 1),
    # bold-wrapped MCQ
    ("**Final answer: (B) bookstore**", "B", 1),
    # ambiguous: "As we discussed" should NOT match
    ("As we discussed earlier, the correct option is (C)", "C", 1),
    # wrong
    ("Final answer: A", "C", 0),
    # multiple candidates: "(D) [...] So the answer is (B)"
    ("(D) is wrong. Therefore, the answer is (B)", "B", 1),
]
for gen, gold, exp in MM:
    got = grade_mmlu(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [mmlu]   {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:80s}")

# ---------- commonsenseqa ----------
CSQA = [
    ("Final answer: (A) bank", "A", 1),
    ("the answer is (E)", "E", 1),
]
for gen, gold, exp in CSQA:
    got = grade_commonsenseqa(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [csqa]   {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:60s}")

# ---------- belebele ----------
BB = [
    # gold='1' (= letter A)
    ("Final answer: (A)", "1", 1),
    # gold='2' (= letter B); model says A => wrong
    ("Final answer: (A)", "2", 0),
    # gold='3' (= letter C); model says C => correct
    ("Answer: C", "3", 1),
]
for gen, gold, exp in BB:
    got = grade_belebele(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [bel]    {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:60s}")

# ---------- fava (yes/no) ----------
F = [
    # qwen-style: "Final answer:" at end, lowercase
    ("Step-by-step ... Final answer: no", "no", 1),
    ("Step-by-step ... Final answer: yes", "yes", 1),
    # mistral-style: "Final answer:" at start, capitalized
    ("Final answer: No\n\nThe passage does not contain hallucination.", "no", 1),
    ("Final answer: Yes\n\nIt has hallucinations.", "yes", 1),
    # wrong
    ("Final answer: yes", "no", 0),
    ("Final answer: No", "yes", 0),
    # bold-wrapped
    ("**Final answer: no**", "no", 1),
    # trailing punctuation
    ("Final answer: no.", "no", 1),
    # no anchor
    ("Step 1: ... Step 2: ...", "no", 0),
    # gold case-insensitive
    ("Final answer: NO", "no", 1),
]
for gen, gold, exp in F:
    got = grade_fava(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [fava]   {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:80s}")

# ---------- ragtruth (same as fava) ----------
R = [
    ("Final answer: no", "no", 1),
    ("Final answer: yes", "yes", 1),
    ("Final answer: no", "yes", 0),
]
for gen, gold, exp in R:
    got = grade_ragtruth(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [rag]    {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:60s}")

# ---------- common_claim_3class (True/False/Neither) ----------
CC = [
    # standard
    ("Step ... Final answer: True", "True", 1),
    ("Step ... Final answer: False", "False", 1),
    ("Step ... Final answer: Neither", "Neither", 1),
    # mistral leading
    ("Final answer: Neither\n\nThe claim does not provide enough info.", "Neither", 1),
    # case-insensitive
    ("Final answer: true", "True", 1),
    ("Final answer: NEITHER", "Neither", 1),
    # wrong
    ("Final answer: True", "False", 0),
    ("Final answer: Neither", "True", 0),
    # bold-wrapped
    ("**Final answer: False**", "False", 1),
    # no anchor
    ("Let me think about this claim...", "True", 0),
]
for gen, gold, exp in CC:
    got = grade_common_claim_3class(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [cc3]    {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:80s}")

# ---------- when2call_3class (A/B/C) ----------
W = [
    # standard end placement
    ("Step 1 ... Final answer: A", "A", 1),
    ("Step 1 ... Final answer: B", "B", 1),
    ("Step 1 ... Final answer: C", "C", 1),
    # paren-wrapped
    ("Final answer: (C) Ask the user for clarification", "C", 1),
    # mistral leading + explanation after
    ("Final answer: A\n\nThe tool can handle this.", "A", 1),
    # wrong
    ("Final answer: A", "B", 0),
    ("Final answer: C", "A", 0),
    # case-insensitive on input but gold is uppercase
    ("Final answer: a", "A", 1),
    # bold-wrapped
    ("**Final answer: B**", "B", 1),
    # invalid letter (D) — model hallucinated a non-ABC choice
    ("Final answer: D", "A", 0),
    # no anchor
    ("I think we should call a tool.", "A", 0),
]
for gen, gold, exp in W:
    got = grade_when2call_3class(gen, gold)
    status = "OK" if got == exp else "FAIL"
    print(f"  [w2c]    {status}  gold={gold!r:8s} pred_label={got} exp={exp}  gen={gen!r:80s}")

print("\n=== extract_number sanity ===")
for s, exp in [
    ("**Final answer: 21**", "21"),
    ("Final answer: $202,650", "202650"),
    ("\\boxed{3}", "3"),
    ("Final answer: 9.", "9"),
]:
    got = extract_number(s)
    status = "OK" if got is not None and got.replace(",","") == exp else "FAIL"
    print(f"  {status}  in={s!r:40s}  got={got!r}  exp={exp!r}")

print("\n=== extract_letter sanity ===")
for s, valid, exp in [
    ("**Final answer: (B) bookstore**", "ABCDE", "B"),
    ("As we discussed", "ABCDE", None),
    ("the correct option is (C)", "ABCDE", "C"),
]:
    got = extract_letter(s, valid)
    status = "OK" if got == exp else "FAIL"
    print(f"  {status}  in={s!r:50s}  got={got!r}  exp={exp!r}")
