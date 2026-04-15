"""Shared matplotlib style for paper figures."""
import matplotlib

COLORS = [
    '#4C72B0',  # steel blue
    '#DD8452',  # warm orange
    '#55A868',  # sage green
    '#C44E52',  # muted red
    '#937DC2',  # purple (5th)
    '#8C8C8C',  # gray (6th)
]
MARKERS = ['o', 's', '^', 'D', 'v', '*']

# 5-dataset palette (consistent across figs)
DATASET_COLORS = {
    "common_claim_3class": '#4C72B0',
    "e2h_amc_3class":      '#DD8452',
    "e2h_amc_5class":      '#55A868',
    "when2call_3class":    '#C44E52',
    "ragtruth_binary":     '#937DC2',
    "fava_binary":         '#8C8C8C',
}

DATASET_LABELS = {
    "common_claim_3class": "CommonClaim",
    "e2h_amc_3class":      "E2H-AMC (3c)",
    "e2h_amc_5class":      "E2H-AMC (5c)",
    "when2call_3class":    "When2Call",
    "ragtruth_binary":     "RAGTruth",
    "fava_binary":         "FAVA",
}

METHOD_LABELS = {
    "lr_probe":       "LR Probe",
    "pca_lr":         "PCA+LR",
    "iti":            "ITI",
    "kb_mlp":         "KB MLP",
    "attn_satisfies": "AttnSat",
    "sep":            "SEP",
    "step":           "STEP",
    "mm_probe":       "MM Probe",
    "lid":            "LID",
    "llm_check":      "LLM-Check",
    "seakr":          "SeaKR",
    "coe":            "CoE",
}


def setup_style():
    matplotlib.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'lines.linewidth': 2.2,
        'lines.markersize': 8,
        'mathtext.fontset': 'stix',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.grid': False,
        'text.usetex': False,
    })
