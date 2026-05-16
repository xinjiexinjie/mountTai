# seedRobustness.py

import os
import glob
import pandas as pd

MAIN_CODES = [
    "Phatic / Ambient Participation",
    "Ritualized Enactment",
    "Lifestyle Translation",
    "Affective Compression",
    "Interpretive Framing",
    "Ironic Distancing",
    "Governance & Official Discourse",
    "Uncoded",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

CODED_FILE = os.path.join(BASE_DIR, "coded_all_platforms.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "seed_level_robustness_douyin_xhs_only.csv")

CODE_COL = "main_code"
PLATFORM_COL = "platform"

# 只对有 seed ID 的平台做稳健性检验
RAW_FILE_PATTERNS = [
    {
        "platform": "xiaohongshu",
        "keyword": "小红书",
    },
    {
        "platform": "douyin",
        "keyword": "抖音",
    },
]

SEED_CANDIDATES = [
    "seed_id",
    "video_id",
    "post_id",
    "aweme_id",
    "note_id",
    "bvid",
    "url",
    "source_url",
    "视频链接",
    "笔记链接",
    "作品链接",
    "作品ID",
    "视频ID",
    "笔记ID",
    "aweme_id",
    "noteId",
    "note_id",
    "item_id",
    "itemId",
    "作品id",
    "视频id",
    "笔记id",
]

COMMENT_ID_CANDIDATES = [
    "评论ID",
    "comment_id",
    "commentId",
    "cid",
]


def normalize_platform(x):
    x = str(x).lower()

    if "douyin" in x or "抖音" in x:
        return "douyin"

    if "xiaohongshu" in x or "xhs" in x or "小红书" in x:
        return "xiaohongshu"

    if "bilibili" in x or "b站" in x or "哔哩" in x:
        return "bilibili"

    return x.strip()


def find_file_by_keyword(base_dir: str, keyword: str):
    patterns = [
        os.path.join(base_dir, f"*{keyword}*.csv"),
        os.path.join(base_dir, f"*{keyword}*.xlsx"),
        os.path.join(base_dir, f"*{keyword}*.xls"),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    matches = [
        m for m in matches
        if "coded_all_platforms" not in os.path.basename(m)
        and "seed_level_robustness" not in os.path.basename(m)
    ]

    if not matches:
        print(f"\nNo raw file found for keyword: {keyword}")
        return None

    if len(matches) > 1:
        print(f"\nMultiple files found for keyword [{keyword}], using the first one:")
        for m in matches:
            print(m)

    return matches[0]


def read_file(fp: str) -> pd.DataFrame:
    ext = os.path.splitext(fp)[1].lower()

    if ext == ".csv":
        return pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(fp)
    else:
        raise ValueError(f"Unsupported file type: {fp}")


def find_seed_col(df: pd.DataFrame):
    for col in SEED_CANDIDATES:
        if col in df.columns:
            return col, "real_seed"

    for col in COMMENT_ID_CANDIDATES:
        if col in df.columns:
            return col, "comment_id_not_recommended"

    return None, "no_seed"


def load_coded_file():
    if not os.path.exists(CODED_FILE):
        raise FileNotFoundError(f"Cannot find coded file: {CODED_FILE}")

    coded = pd.read_csv(CODED_FILE, encoding="utf-8-sig", low_memory=False)

    print("Coded file columns:")
    print(list(coded.columns))

    if PLATFORM_COL not in coded.columns:
        raise ValueError(f"coded_all_platforms.csv does not contain column: {PLATFORM_COL}")

    if CODE_COL not in coded.columns:
        raise ValueError(f"coded_all_platforms.csv does not contain column: {CODE_COL}")

    coded["platform_norm"] = coded[PLATFORM_COL].apply(normalize_platform)
    coded["main_code_for_robustness"] = coded[CODE_COL].astype(str).str.strip()

    # 只保留抖音和小红书
    coded = coded[coded["platform_norm"].isin(["douyin", "xiaohongshu"])].copy()

    # 为了和原始文件按平台内部行序对应
    coded["_row_no_within_platform"] = coded.groupby("platform_norm").cumcount()

    return coded


def load_raw_seed_info():
    raw_frames = []

    for item in RAW_FILE_PATTERNS:
        platform = item["platform"]
        keyword = item["keyword"]

        fp = find_file_by_keyword(BASE_DIR, keyword)

        if fp is None:
            print(f"Warning: No raw file found for {platform}.")
            continue

        print(f"\nReading raw file for {platform}: {fp}")

        raw = read_file(fp)

        print("Raw columns:")
        print(list(raw.columns))

        seed_col, seed_type = find_seed_col(raw)

        if seed_col is None:
            print(f"Warning: No seed column found for {platform}. This platform will be skipped.")
            continue

        if seed_type != "real_seed":
            print(
                f"Warning: {seed_col} looks like a comment-level ID, not a seed-level ID. "
                f"This platform will be skipped."
            )
            continue

        print(f"Using seed column for {platform}: {seed_col} ({seed_type})")

        raw_seed = pd.DataFrame({
            "platform_norm": [platform] * len(raw),
            "_row_no_within_platform": range(len(raw)),
            "seed_id_from_raw": platform + "__" + raw[seed_col].astype(str),
            "seed_type_used": [seed_type] * len(raw),
            "raw_source_file": [os.path.basename(fp)] * len(raw),
        })

        raw_frames.append(raw_seed)

    if not raw_frames:
        raise ValueError(
            "No usable raw seed information found for Douyin or Xiaohongshu. "
            "Please check whether the raw files contain video_id, note_id, post_id, url, or similar columns."
        )

    return pd.concat(raw_frames, ignore_index=True)


def attach_seed_to_coded(coded: pd.DataFrame, raw_seed: pd.DataFrame):
    merged = coded.merge(
        raw_seed,
        on=["platform_norm", "_row_no_within_platform"],
        how="left"
    )

    # 只保留成功匹配到 seed 的行
    before = len(merged)
    merged = merged.dropna(subset=["seed_id_from_raw"]).copy()
    after = len(merged)

    print(f"\nRows before seed matching: {before}")
    print(f"Rows after seed matching: {after}")
    print(f"Rows dropped because seed was missing: {before - after}")

    merged["seed_id_for_robustness"] = merged["seed_id_from_raw"]

    return merged


def main():
    print("BASE_DIR:", BASE_DIR)

    coded = load_coded_file()
    raw_seed = load_raw_seed_info()

    df = attach_seed_to_coded(coded, raw_seed)

    print("\nPlatform counts after excluding Bilibili:")
    print(df["platform_norm"].value_counts())

    print("\nSeed type used:")
    print(df[["platform_norm", "seed_type_used", "raw_source_file"]].drop_duplicates())

    # 只保留主编码类别
    df = df[df["main_code_for_robustness"].isin(MAIN_CODES)].copy()

    if df.empty:
        raise ValueError(
            "After filtering by MAIN_CODES, no rows remain.\n"
            "Please check whether main_code labels match MAIN_CODES."
        )

    PLATFORM = "platform_norm"
    CODE = "main_code_for_robustness"

    # comment-level shares:
    # 在抖音和小红书内部，每条评论同等权重
    comment_counts = (
        df.groupby([PLATFORM, CODE])
        .size()
        .reset_index(name="n_comment_level")
    )

    comment_counts["share_comment_level"] = (
        comment_counts["n_comment_level"]
        / comment_counts.groupby(PLATFORM)["n_comment_level"].transform("sum")
    )

    # seed-level shares:
    # 每个作品/笔记同等权重
    seed_counts = (
        df.groupby([PLATFORM, "seed_id_for_robustness", CODE])
        .size()
        .reset_index(name="n")
    )

    seed_totals = (
        df.groupby([PLATFORM, "seed_id_for_robustness"])
        .size()
        .reset_index(name="seed_total")
    )

    # 关键修正：
    # 需要补齐每个 seed 在每个 MAIN_CODE 上的 0 值。
    # 否则只会平均“出现过该编码的 seed”，导致 share_seed_level_mean 被高估。
    all_seeds = df[[PLATFORM, "seed_id_for_robustness"]].drop_duplicates()
    all_codes = pd.DataFrame({CODE: MAIN_CODES})

    complete_grid = (
        all_seeds.assign(_key=1)
        .merge(all_codes.assign(_key=1), on="_key")
        .drop(columns="_key")
    )

    seed_counts_complete = complete_grid.merge(
        seed_counts,
        on=[PLATFORM, "seed_id_for_robustness", CODE],
        how="left"
    )

    seed_counts_complete["n"] = seed_counts_complete["n"].fillna(0)

    seed_counts_complete = seed_counts_complete.merge(
        seed_totals,
        on=[PLATFORM, "seed_id_for_robustness"],
        how="left"
    )

    seed_counts_complete["seed_share"] = (
        seed_counts_complete["n"] / seed_counts_complete["seed_total"]
    )

    seed_level = (
        seed_counts_complete.groupby([PLATFORM, CODE])["seed_share"]
        .mean()
        .reset_index(name="share_seed_level_mean")
    )

    out = comment_counts.merge(
        seed_level,
        on=[PLATFORM, CODE],
        how="outer"
    )

    out["share_comment_level"] = out["share_comment_level"].fillna(0)
    out["share_seed_level_mean"] = out["share_seed_level_mean"].fillna(0)
    out["n_comment_level"] = out["n_comment_level"].fillna(0).astype(int)

    out["abs_difference"] = (
        out["share_comment_level"] - out["share_seed_level_mean"]
    ).abs()

    out = out.sort_values([PLATFORM, CODE])

    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(out)

    print("\nCheck: comment-level shares should sum to 1 within each platform:")
    print(out.groupby(PLATFORM)["share_comment_level"].sum())

    print("\nCheck: seed-level shares should also sum to 1 within each platform:")
    print(out.groupby(PLATFORM)["share_seed_level_mean"].sum())

    print("\nNumber of seeds by platform:")
    print(
        df.groupby(PLATFORM)["seed_id_for_robustness"]
        .nunique()
        .reset_index(name="n_seeds")
    )

    print("\nNote:")
    print(
        "Bilibili was excluded from the seed-level robustness check because the danmu raw file does not contain a reliable video-level seed identifier."
    )


if __name__ == "__main__":
    main()