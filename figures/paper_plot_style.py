"""Shared matplotlib style for paper figures.

Palette is a curated NeurIPS-leaning set (Okabe-Ito-adjacent, colorblind-safe,
print-robust). Fonts are bumped high enough to stay readable when figures are
inserted at 0.75-1.0 linewidth in a two-column layout.
"""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

COLORS = [
    '#2E5EAA',  # deep navy
    '#E07A3E',  # burnt amber
    '#3A8A5A',  # forest green
    '#B8384E',  # clay red
    '#6E52A3',  # aubergine
    '#4A4A4A',  # charcoal
]
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

# Per-signal-family colors (used in clustering & as accents)
FAMILY_COLORS = {
    'hidden':     '#2E5EAA',
    'attention':  '#E07A3E',
    'generation': '#3A8A5A',
}

# 5-dataset palette (consistent across figs)
DATASET_COLORS = {
    "common_claim_3class": '#2E5EAA',
    "e2h_amc_3class":      '#E07A3E',
    "e2h_amc_5class":      '#3A8A5A',
    "when2call_3class":    '#B8384E',
    "ragtruth_binary":     '#6E52A3',
    "fava_binary":         '#4A4A4A',
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


# Custom sequential colormap: soft cream -> deep navy.
# Reads calm and publication-grade; avoids the saturated red of YlOrRd.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'paper_seq',
    [
        (0.00, '#F5EFE6'),   # parchment
        (0.25, '#C9D6E8'),   # pale blue
        (0.55, '#6B90C6'),   # mid blue
        (0.80, '#2E5EAA'),   # deep navy
        (1.00, '#0E2C5E'),   # midnight
    ],
)

# Diverging map (neutral cream center, navy negative, amber positive) —
# more tasteful than RdBu_r for percentage-point contribution displays.
CMAP_DIV = LinearSegmentedColormap.from_list(
    'paper_div',
    [
        (0.00, '#0E2C5E'),
        (0.25, '#6B90C6'),
        (0.50, '#F5EFE6'),
        (0.75, '#E0A86A'),
        (1.00, '#8F3A14'),
    ],
)

# Correlation-oriented sequential map (low=parchment, high=forest).
CMAP_CORR = LinearSegmentedColormap.from_list(
    'paper_corr',
    [
        (0.00, '#F5EFE6'),
        (0.35, '#A7C4A0'),
        (0.70, '#3A8A5A'),
        (1.00, '#123F24'),
    ],
)


def setup_style():
    # Sizes calibrated for figures saved at ~6.5-7 in and inserted at
    # \linewidth (5.5 in in NeurIPS single-column). At those dimensions
    # 9-10 pt figure text renders near 8 pt on paper, matching the body.
    matplotlib.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'regular',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
        'lines.linewidth': 1.6,
        'lines.markersize': 5.5,
        'mathtext.fontset': 'stix',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.9,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'xtick.color': '#1a1a1a',
        'ytick.color': '#1a1a1a',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.grid': False,
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
