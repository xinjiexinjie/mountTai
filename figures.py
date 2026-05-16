import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =========================
# 0) INPUT (single file)
# =========================
PATH = "coding_summary_updated_from_uncoded_suggestions_FIXED.csv"
OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# Column name autodetect
# -------------------------
def pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

df0 = pd.read_csv(PATH)

COL_PLATFORM = pick_col(df0.columns, ["platform", "Platform"])
COL_MAIN     = pick_col(df0.columns, ["main_code", "Main code", "Main_code", "code", "main"])
COL_SUB      = pick_col(df0.columns, ["sub_code", "Sub-code", "Sub_code", "subcode", "sub"])
COL_N        = pick_col(df0.columns, ["n", "N", "count", "freq", "frequency"])

if not all([COL_PLATFORM, COL_MAIN, COL_N]):
    raise ValueError(
        f"Could not detect required columns. Detected: "
        f"platform={COL_PLATFORM}, main={COL_MAIN}, n={COL_N}, sub={COL_SUB}. "
        f"Please rename columns or set COL_* manually."
    )

# 如果你的 sub_code 列不是空值而是用某个固定字符串表示“无子码”，也可以在这里统一成 NaN
if COL_SUB is None:
    df0["__sub_code__"] = np.nan
    COL_SUB = "__sub_code__"

# =========================
# 1) Orders (match your paper)
# =========================
PLATFORM_ORDER = ["Bilibili", "Xiaohongshu", "Douyin"]

MAIN_ORDER = [
    "Affective Compression",
    "Governance & Official Discourse",
    "Interpretive Framing",
    "Ironic Distancing",
    "Lifestyle Translation",
    "Phatic / Ambient Participation",
    "Ritualized Enactment",
    "Uncoded",
]

RITUAL_SUB_ORDER = [
    "Endurance & Body Test",
    "Night Climb & Threshold",
    "Pilgrimage Language",
    "Sunrise Completion",
    "Vernacular Blessing",
]
PHATIC_SUB_ORDER = [
    "Crowd Presence",
    "Interaction Maintenance",
    "Minimal Response",
]
LIFESTYLE_SUB_ORDER = [
    "Aesthetic Display",
    "Consumption Framing",
    "Itinerary Optimization",
    "Self-Branding",
]
AFFECTIVE_SUB_ORDER = [
    "Awe & Sublimity",
    "Exhaustion + Worth",
    "Healing / Reset",
    "Pride / Achievement",
]

# =========================
# 2) Pastel palettes (close to your screenshot)
# =========================
PALETTE_MAIN = {
    "Affective Compression": "#7FCDBB",
    "Governance & Official Discourse": "#FC8D62",
    "Interpretive Framing": "#8DA0CB",
    "Ironic Distancing": "#E78AC3",
    "Lifestyle Translation": "#A6D854",
    "Phatic / Ambient Participation": "#FFD92F",
    "Ritualized Enactment": "#E5C494",
    "Uncoded": "#B3B3B3",
}

PALETTE_AFFECTIVE = {
    "Awe & Sublimity": "#7FCDBB",
    "Exhaustion + Worth": "#FFFFB3",
    "Healing / Reset": "#BEBADA",
    "Pride / Achievement": "#FB8072",
}
PALETTE_LIFESTYLE = {
    "Aesthetic Display": "#7FCDBB",
    "Consumption Framing": "#FFFFB3",
    "Itinerary Optimization": "#BEBADA",
    "Self-Branding": "#FB8072",
}
PALETTE_PHATIC = {
    "Crowd Presence": "#7FCDBB",
    "Interaction Maintenance": "#FFFFB3",
    "Minimal Response": "#BEBADA",
}
PALETTE_RITUAL = {
    "Endurance & Body Test": "#7FCDBB",
    "Night Climb & Threshold": "#FFFFB3",
    "Pilgrimage Language": "#BEBADA",
    "Sunrise Completion": "#FB8072",
    "Vernacular Blessing": "#80B1D3",
}

# =========================
# 3) Plot helper: 100% stacked horizontal bar
# =========================
def plot_100_stacked_hbar(df_share: pd.DataFrame,
                         category_order: list,
                         color_map: dict,
                         xlabel: str,
                         outfile: str,
                         platform_order: list = PLATFORM_ORDER,
                         legend_ncol: int = 4):
    df = df_share.copy()
    df = df.reindex(platform_order)
    df = df.reindex(columns=category_order).fillna(0.0)

    fig_h = 2.6 + 0.55 * len(platform_order)
    fig_w = 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    left = np.zeros(len(df))
    y = np.arange(len(df.index))

    for cat in df.columns:
        vals = df[cat].values
        if np.allclose(vals, 0):
            continue
        ax.barh(
            y, vals, left=left, height=0.78,
            color=color_map.get(cat, None),
            edgecolor="white", linewidth=1.0,
            label=cat
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title("")  # no title

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(legend_ncol, len(labels)),
            frameon=False,
            fontsize=11
        )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================
# 4) Build shares
# =========================
def main_code_shares_from_single_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some rows may be subcode rows; we ignore sub_code and aggregate to main_code level.
    """
    tmp = df[[COL_PLATFORM, COL_MAIN, COL_N]].copy()
    tmp[COL_N] = pd.to_numeric(tmp[COL_N], errors="coerce").fillna(0)

    agg = tmp.groupby([COL_PLATFORM, COL_MAIN], as_index=False)[COL_N].sum()

    totals = agg.groupby(COL_PLATFORM, as_index=False)[COL_N].sum().rename(columns={COL_N: "platform_total"})
    agg = agg.merge(totals, on=COL_PLATFORM, how="left")
    agg["share"] = np.where(agg["platform_total"] > 0, agg[COL_N] / agg["platform_total"], 0.0)

    pivot = agg.pivot_table(index=COL_PLATFORM, columns=COL_MAIN, values="share", aggfunc="sum").fillna(0.0)
    return pivot

def subcode_shares_within_main(df: pd.DataFrame, main_code_value: str) -> pd.DataFrame:
    """
    Use rows where main_code matches and sub_code is not null.
    Normalize within (platform, main_code) so subcodes sum to 1 per platform.
    """
    tmp = df[[COL_PLATFORM, COL_MAIN, COL_SUB, COL_N]].copy()
    tmp[COL_N] = pd.to_numeric(tmp[COL_N], errors="coerce").fillna(0)

    tmp = tmp[tmp[COL_MAIN] == main_code_value].copy()
    tmp = tmp[~tmp[COL_SUB].isna()].copy()

    if tmp.empty:
        raise ValueError(f"No subcode rows found for main_code='{main_code_value}'. "
                         f"Check that your file has a sub_code column and values for this main code.")

    denom = tmp.groupby(COL_PLATFORM, as_index=False)[COL_N].sum().rename(columns={COL_N: "den"})
    tmp = tmp.merge(denom, on=COL_PLATFORM, how="left")
    tmp["share"] = np.where(tmp["den"] > 0, tmp[COL_N] / tmp["den"], 0.0)

    pivot = tmp.pivot_table(index=COL_PLATFORM, columns=COL_SUB, values="share", aggfunc="sum").fillna(0.0)
    return pivot

# =========================
# 5) Generate 5 figures
# =========================
# Fig 1: main code
main_share = main_code_shares_from_single_file(df0)
plot_100_stacked_hbar(
    df_share=main_share,
    category_order=MAIN_ORDER,
    color_map=PALETTE_MAIN,
    xlabel="Share within platform",
    outfile=os.path.join(OUTDIR, "fig_maincode_stacked.png"),
    legend_ncol=4
)

# Fig 2–5: subcodes
ritual_share = subcode_shares_within_main(df0, "Ritualized Enactment").reindex(PLATFORM_ORDER).reindex(columns=RITUAL_SUB_ORDER)
plot_100_stacked_hbar(
    df_share=ritual_share,
    category_order=RITUAL_SUB_ORDER,
    color_map=PALETTE_RITUAL,
    xlabel="Subcode share within Ritualized Enactment",
    outfile=os.path.join(OUTDIR, "fig_ritual_subcodes.png"),
    legend_ncol=3
)

phatic_share = subcode_shares_within_main(df0, "Phatic / Ambient Participation").reindex(PLATFORM_ORDER).reindex(columns=PHATIC_SUB_ORDER)
plot_100_stacked_hbar(
    df_share=phatic_share,
    category_order=PHATIC_SUB_ORDER,
    color_map=PALETTE_PHATIC,
    xlabel="Subcode share within Phatic / Ambient Participation",
    outfile=os.path.join(OUTDIR, "fig_phatic_subcodes.png"),
    legend_ncol=3
)

lifestyle_share = subcode_shares_within_main(df0, "Lifestyle Translation").reindex(PLATFORM_ORDER).reindex(columns=LIFESTYLE_SUB_ORDER)
plot_100_stacked_hbar(
    df_share=lifestyle_share,
    category_order=LIFESTYLE_SUB_ORDER,
    color_map=PALETTE_LIFESTYLE,
    xlabel="Subcode share within Lifestyle Translation",
    outfile=os.path.join(OUTDIR, "fig_lifestyle_subcodes.png"),
    legend_ncol=3
)

affective_share = subcode_shares_within_main(df0, "Affective Compression").reindex(PLATFORM_ORDER).reindex(columns=AFFECTIVE_SUB_ORDER)
plot_100_stacked_hbar(
    df_share=affective_share,
    category_order=AFFECTIVE_SUB_ORDER,
    color_map=PALETTE_AFFECTIVE,
    xlabel="Subcode share within Affective Compression",
    outfile=os.path.join(OUTDIR, "fig_affective_subcodes.png"),
    legend_ncol=4
)

print("Saved figures to:", OUTDIR)