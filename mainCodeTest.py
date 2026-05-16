import os
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations

try:
    from statsmodels.stats.multitest import multipletests
    HAS_SM = True
except Exception:
    HAS_SM = False


# =========================================================
# 0) Load
# =========================================================
FULL_FILE = "coded_all_platforms.csv"
RECODE_FILE = "uncoded_diagnosis_with_suggestions.csv"

OUTDIR = "stats_outputs"
os.makedirs(OUTDIR, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    # 避免 mixed types 的 DtypeWarning（不影响结果，但会烦）
    return pd.read_csv(path, low_memory=False)


MAIN_CODES_CANON = [
    "Phatic / Ambient Participation",
    "Ritualized Enactment",
    "Lifestyle Translation",
    "Affective Compression",
    "Interpretive Framing",
    "Ironic Distancing",
    "Governance & Official Discourse",
    "Uncoded",
]

# 一些常见的“列名候选”（如果命中了就直接用，省得扫描）
MAIN_CODE_NAME_CANDIDATES = [
    "dominant_main_code_updated", "dominant_main_code_adjusted", "dominant_main_code_final",
    "main_code_updated", "main_code_adjusted", "main_code_final", "final_main_code",
    "suggested_main_code", "updated_main_code", "new_main_code",
    "dominant_main_code", "main_code", "主码", "主码建议", "建议主码", "主码_建议", "主码_更新", "主码_最终"
]

# =========================================================
# 1) Column detection helpers
# =========================================================
def guess_col(df: pd.DataFrame, candidates):
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def guess_platform_col(df: pd.DataFrame) -> str:
    col = guess_col(df, ["platform", "platform_hint", "平台", "source", "site"])
    if col is None:
        raise ValueError("Cannot find platform column (expected platform/platform_hint/平台/source/site).")
    return col


def guess_text_col(df: pd.DataFrame) -> str:
    # Prefer already-normalized text if present
    preferred = ["text_norm_600", "text_norm", "text_norm600", "text_600"]
    col = guess_col(df, preferred)
    if col is not None:
        return col

    fallbacks = ["text", "content", "comment", "message", "评论", "弹幕", "danmu", "正文", "内容", "留言", "评论内容", "弹幕内容"]
    col = guess_col(df, fallbacks)
    if col is not None:
        return col

    raise ValueError("Cannot find text column. Expected text_norm_600/text/comment/content/评论/弹幕 etc.")

def guess_main_code_col(df: pd.DataFrame) -> str:
    # 1) 先按列名快速匹配
    cols = {str(c).lower(): c for c in df.columns}
    for cand in MAIN_CODE_NAME_CANDIDATES:
        if cand.lower() in cols:
            return cols[cand.lower()]

    # 2) 列名没命中 -> 按取值扫描：看哪列像“主码列”
    #    规则：列中属于 MAIN_CODES_CANON 的占比越高越像主码列
    best_col, best_score = None, -1.0

    # 为了速度：最多抽样 50000 行评估
    n = len(df)
    sample_df = df if n <= 50000 else df.sample(50000, random_state=42)

    canon_set = set([c.lower() for c in MAIN_CODES_CANON])

    for c in sample_df.columns:
        # 只看 object/string 类列
        if sample_df[c].dtype != object:
            continue

        s = sample_df[c].astype(str).str.strip()
        s = s[s.ne("") & s.str.lower().ne("nan")]
        if len(s) == 0:
            continue

        # 计算命中主码集合的比例
        hit = s.str.lower().isin(canon_set).mean()

        # 同时奖励“不同取值数量接近主码数量”的列（避免把 platform/其他列误判）
        nunique = s.nunique()
        uniq_bonus = 1.0 - min(abs(nunique - len(MAIN_CODES_CANON)) / len(MAIN_CODES_CANON), 1.0)  # 0~1

        score = 0.85 * hit + 0.15 * uniq_bonus

        if score > best_score:
            best_score, best_col = score, c

    if best_col is None or best_score < 0.20:
        # 给出可读的诊断信息，方便你一眼看列名
        preview = list(df.columns)[:60]
        raise ValueError(
            "Cannot find a main-code column by name or by value.\n"
            f"Top columns preview (first 60): {preview}\n"
            "Tip: open the recoded CSV and check which column contains values like "
            f"{MAIN_CODES_CANON}."
        )

    print(f"[Auto-detected main-code column] '{best_col}' (score={best_score:.3f})")
    return best_col


def normalize_text(x: str, max_len: int = 600) -> str:
    s = re.sub(r"\s+", " ", str(x)).strip()
    return s[:max_len]


def find_best_key(full_df: pd.DataFrame, rec_df: pd.DataFrame):
    """
    1) Try shared ID-like columns first.
    2) Otherwise fall back to composite key: platform + normalized text.
    """
    id_like = [
        "row_id", "orig_row_id", "original_row_id", "orig_index", "original_index",
        "index", "idx", "id", "uid", "case_id"
    ]
    full_cols = {c.lower(): c for c in full_df.columns}
    rec_cols = {c.lower(): c for c in rec_df.columns}

    shared = []
    for c in id_like:
        if c in full_cols and c in rec_cols:
            shared.append((full_cols[c], rec_cols[c]))

    # Choose the shared column that is most unique in recoded file
    best = None
    best_uni = -1
    for fcol, rcol in shared:
        uni = rec_df[rcol].nunique(dropna=True)
        if uni > best_uni:
            best_uni = uni
            best = (fcol, rcol)

    if best is not None and best_uni > 0:
        return ("shared_id", best[0], best[1])

    return ("platform_text", None, None)


# =========================================================
# 2) Build adjusted full corpus by backfilling Uncoded
# =========================================================
full = read_csv(FULL_FILE)
rec = read_csv(RECODE_FILE)

full_platform_col = guess_platform_col(full)
full_text_col = guess_text_col(full)
full_code_col = guess_main_code_col(full)  # should be dominant_main_code in your full file

rec_platform_col = guess_platform_col(rec)
rec_text_col = guess_text_col(rec)
rec_code_col = guess_main_code_col(rec)

# Prepare normalized text columns for safe matching
full["_platform_"] = full[full_platform_col].astype(str).str.strip()
rec["_platform_"] = rec[rec_platform_col].astype(str).str.strip()
full["_text_norm_"] = full[full_text_col].astype(str).apply(normalize_text)
rec["_text_norm_"] = rec[rec_text_col].astype(str).apply(normalize_text)

# Identify key strategy
mode, f_idcol, r_idcol = find_best_key(full, rec)

print(f"[Match mode] {mode}")
if mode == "shared_id":
    print(f"Using shared ID columns: full.{f_idcol} <-> rec.{r_idcol}")
else:
    print("No reliable shared ID column found. Falling back to composite key: platform + normalized text.")

# Filter to the uncoded subset in the full file
mask_uncoded = full[full_code_col].astype(str).str.strip().eq("Uncoded")
full_uncoded = full.loc[mask_uncoded].copy()
n_uncoded_full = len(full_uncoded)
print(f"[Full corpus] rows={len(full):,}; Uncoded rows={n_uncoded_full:,}")

# Keep only recoded rows with a non-empty main code
rec_main = rec[rec_code_col].astype(str).str.strip()
rec_valid = rec.loc[rec_main.ne("") & rec_main.str.lower().ne("nan")].copy()
print(f"[Recoded file] rows={len(rec):,}; rows with valid main code={len(rec_valid):,}")

# Build mapping
if mode == "shared_id":
    # Validate uniqueness in recoded file
    dup = rec_valid[r_idcol].duplicated(keep=False)
    if dup.any():
        dups = rec_valid.loc[dup, r_idcol].head(10).tolist()
        print(f"WARNING: recoded ID column '{r_idcol}' has duplicates. Example IDs: {dups}")
        print("Will keep the first occurrence per ID.")
    rec_map = rec_valid.drop_duplicates(subset=[r_idcol]).set_index(r_idcol)[rec_code_col].to_dict()

    # Apply mapping: only to uncoded rows
    def map_code(row):
        key = row.get(f_idcol, None)
        if pd.isna(key):
            return None
        return rec_map.get(key, None)

    full["_recoded_main_"] = None
    full.loc[mask_uncoded, "_recoded_main_"] = full.loc[mask_uncoded].apply(map_code, axis=1)

else:
    # Composite key
    rec_valid["_key_"] = rec_valid["_platform_"] + "||" + rec_valid["_text_norm_"]
    full["_key_"] = full["_platform_"] + "||" + full["_text_norm_"]

    # If duplicates exist in recoded keys, keep first and warn
    dup = rec_valid["_key_"].duplicated(keep=False)
    if dup.any():
        print("WARNING: duplicate composite keys in recoded file (platform+text).")
        print("This usually means identical short texts appear multiple times; keeping first per key.")
        print(rec_valid.loc[dup, ["_platform_", "_text_norm_"]].head(5).to_string(index=False))

    rec_map = rec_valid.drop_duplicates(subset=["_key_"]).set_index("_key_")[rec_code_col].to_dict()

    full["_recoded_main_"] = None
    full.loc[mask_uncoded, "_recoded_main_"] = full.loc[mask_uncoded, "_key_"].map(rec_map)

# Create adjusted main code
full["dominant_main_code_original"] = full[full_code_col].astype(str).str.strip()
full["dominant_main_code_adjusted"] = full["dominant_main_code_original"]

# Replace only where original is Uncoded and recoded value exists
can_replace = mask_uncoded & full["_recoded_main_"].notna() & (full["_recoded_main_"].astype(str).str.strip() != "")
full.loc[can_replace, "dominant_main_code_adjusted"] = full.loc[can_replace, "_recoded_main_"].astype(str).str.strip()

# Diagnostics
n_matched = int(can_replace.sum())
n_uncoded_remaining = int((full["dominant_main_code_adjusted"] == "Uncoded").sum())

print(f"[Backfill] Uncoded rows in full={n_uncoded_full:,}")
print(f"[Backfill] Matched & replaced={n_matched:,} ({(n_matched/n_uncoded_full*100 if n_uncoded_full else 0):.1f}%)")
print(f"[Adjusted] Uncoded remaining in full={n_uncoded_remaining:,} ({(n_uncoded_remaining/len(full)*100):.1f}% of full corpus)")

# Save adjusted full file
ADJUSTED_FULL_FILE = os.path.join(OUTDIR, "coded_all_platforms_adjusted_full.csv")
full.to_csv(ADJUSTED_FULL_FILE, index=False)
print(f"Saved adjusted full corpus to: {ADJUSTED_FULL_FILE}")


# =========================================================
# 3) Inferential test: χ² + Cramér’s V + residuals + pairwise
# =========================================================
def cramers_v(chi2, n, r, k):
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

def standardized_residuals(obs):
    obs = np.asarray(obs, dtype=float)
    n = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    exp = row_sums @ col_sums / n
    rowp = row_sums / n
    colp = col_sums / n
    denom = np.sqrt(exp * (1 - rowp) * (1 - colp))
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (obs - exp) / denom
    return exp, res

def chi_square_maincode(df_long: pd.DataFrame):
    tab = pd.crosstab(df_long["platform"], df_long["main_code"])
    obs = tab.values
    chi2, p, dof, _ = chi2_contingency(obs)
    n = int(obs.sum())
    v = cramers_v(chi2, n, tab.shape[0], tab.shape[1])
    exp, std_res = standardized_residuals(obs)

    share = tab.div(tab.sum(axis=1), axis=0)

    return {
        "contingency": tab,
        "share": share,
        "expected": pd.DataFrame(exp, index=tab.index, columns=tab.columns),
        "std_residuals": pd.DataFrame(std_res, index=tab.index, columns=tab.columns),
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "n": n,
        "cramers_v": v,
    }

def pairwise_platform_tests(df_long: pd.DataFrame):
    platforms = sorted(df_long["platform"].unique())
    rows = []
    for a, b in combinations(platforms, 2):
        sub = df_long[df_long["platform"].isin([a, b])]
        tab = pd.crosstab(sub["platform"], sub["main_code"])
        if tab.shape[0] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(tab.values)
        n = int(tab.values.sum())
        v = cramers_v(chi2, n, tab.shape[0], tab.shape[1])
        rows.append({"platform_A": a, "platform_B": b, "chi2": chi2, "dof": dof, "p_raw": p, "cramers_v": v})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    if HAS_SM:
        reject, p_adj, _, _ = multipletests(out["p_raw"].values, method="holm")
        out["p_holm"] = p_adj
        out["reject_0.05_holm"] = reject
    else:
        out["p_holm"] = np.nan
        out["reject_0.05_holm"] = np.nan
    return out.sort_values("p_raw")

def star_residual(x):
    ax = abs(x)
    if ax >= 3.29:
        return f"{x:.2f}***"
    if ax >= 2.58:
        return f"{x:.2f}**"
    if ax >= 1.96:
        return f"{x:.2f}*"
    return f"{x:.2f}"

def report_sentence(label, s):
    p = s["p"]
    p_str = "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")
    return (f"{label}: χ²({s['dof']}, N={s['n']}) = {s['chi2']:.2f}, "
            f"p {p_str}, Cramér’s V = {s['cramers_v']:.3f}.")

def run_inference(df_full: pd.DataFrame, code_col: str, label: str):
    df_long = pd.DataFrame({
        "platform": df_full[full_platform_col].astype(str).str.strip(),
        "main_code": df_full[code_col].astype(str).str.strip()
    })
    df_long = df_long[(df_long["platform"] != "") & (df_long["main_code"] != "") & (df_long["main_code"].str.lower() != "nan")]

    stats = chi_square_maincode(df_long)
    pairwise = pairwise_platform_tests(df_long)

    # starred residuals
    res_star = stats["std_residuals"].applymap(star_residual)

    print("\n" + "=" * 100)
    print(report_sentence(label, stats))
    print("=" * 100)

    # Save tables
    slug = label.lower().replace(" ", "_").replace("/", "_")
    stats["contingency"].to_csv(os.path.join(OUTDIR, f"{slug}_contingency.csv"))
    stats["share"].to_csv(os.path.join(OUTDIR, f"{slug}_share_within_platform.csv"))
    stats["expected"].to_csv(os.path.join(OUTDIR, f"{slug}_expected.csv"))
    stats["std_residuals"].to_csv(os.path.join(OUTDIR, f"{slug}_std_residuals.csv"))
    res_star.to_csv(os.path.join(OUTDIR, f"{slug}_std_residuals_star.csv"))
    pairwise.to_csv(os.path.join(OUTDIR, f"{slug}_pairwise_platform_tests.csv"), index=False)

    return stats, pairwise, res_star

# Run inference for original vs adjusted
stats_orig, pair_orig, res_orig = run_inference(full, "dominant_main_code_original", "Original full coding")
stats_adj,  pair_adj,  res_adj  = run_inference(full, "dominant_main_code_adjusted", "Adjusted full coding")

print("\nSaved all inferential outputs to ./stats_outputs/")
print("Key files to inspect:")
print(" - original_full_coding_std_residuals_star.csv")
print(" - adjusted_full_coding_std_residuals_star.csv")
print(" - *_pairwise_platform_tests.csv (Holm-adjusted p if statsmodels is installed)")