import os
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError


# =========================================================
# 0) Method-aligned codebook (Main codes + Subcodes)
# =========================================================

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

SUBCODES: Dict[str, List[str]] = {
    "Phatic / Ambient Participation": [
        "Crowd Presence",
        "Interaction Maintenance",
        "Minimal Response",
    ],
    "Ritualized Enactment": [
        "Endurance & Body Test",
        "Night Climb & Threshold",
        "Pilgrimage Language",
        "Sunrise Completion",
        "Vernacular Blessing",
    ],
    "Lifestyle Translation": [
        "Aesthetic Display",
        "Consumption Framing",
        "Itinerary Optimization",
        "Self-Branding",
    ],
    "Affective Compression": [
        "Awe & Sublimity",
        "Exhaustion + Worth",
        "Healing / Reset",
        "Pride / Achievement",
    ],
    "Interpretive Framing": [
        "Cultural Hierarchy",
        "Educational Tone",
        "Historical Reference",
    ],
    "Ironic Distancing": [
        "Crowd Irony",
        "Cynical Observation",
        "Meme Language",
    ],
    "Governance & Official Discourse": [
        "Institutional Voice",
        "Official Promotion",
    ],
    "Uncoded": [],
}

CODEBOOK_TEXT = """\
You are assisting a qualitative study on Mount Tai across Chinese digital platforms.

Task:
Classify each text into participation postures (NOT topical themes). A posture is an interactionally enacted way of positioning toward Mount Tai in platform-mediated environments.

You MUST follow this codebook exactly.

MAIN CODES (choose up to 3 total codes; one must be dominant):
1) Phatic / Ambient Participation:
   Minimal engagement, presence-marking, etiquette, coordination requests; short formulaic signals; emoji-only or near-emoji-only.
2) Ritualized Enactment:
   Ritual/pilgrimage lexicon; night climb, sunrise completion, endurance trial; vernacular blessing/merit formulas.
3) Lifestyle Translation:
   Aesthetic self-presentation, consumption framing, itinerary tips/optimization, self-branding/persona.
4) Affective Compression:
   Compressed affective evaluation: awe/sublime, exhaustion+worth, healing/reset, pride/achievement.
5) Interpretive Framing:
   Explanatory/didactic stance; historical/literary references; cultural hierarchy/legitimacy claims.
6) Ironic Distancing:
   Meme-like or playful ridicule; cynical stance; irony about crowds/congestion; distancing via teasing.
7) Governance & Official Discourse:
   Institutional/policy register; official promotional tone; '文旅/传承/正能量/高质量发展' etc.
8) Uncoded:
   Residual for noise/unclassifiable strings after careful attempt (garbled text, isolated numerals, context-insufficient fragments).
   Use sparingly and conservatively; do NOT force interpretation.

SUBCODES:
For each assigned main code (except Uncoded), choose exactly ONE subcode from the allowed list for that main code.

Boundary rules (recurring decision points):
- Ritual shorthand + affect co-occurrence: if ritual formula performs blessing/merit request (e.g., '接好运/保佑/许愿/还愿/泰山奶奶/碧霞祠'), prioritize Ritualized Enactment; affect words may be secondary.
- Meme-like critique vs negative affect: if stance is distancing through meme/irony, prioritize Ironic Distancing rather than Affective Compression.

Output requirements:
Return ONLY JSON following the schema.
- Provide: dominant_main_code, dominant_sub_code, all_codes (up to 3), evidence (quote exact words), confidence, notes.
- Evidence must be short verbatim snippets from the original text (Chinese allowed).
- If Uncoded, dominant_sub_code must be "" and evidence should explain why it's unclassifiable (still quoting the string if possible).
Be neutral and descriptive. Do not invent new categories.
"""


# =========================================================
# 1) JSON Schema (strict Structured Outputs)
# =========================================================
CODING_SCHEMA: Dict[str, Any] = {
    "name": "mount_tai_participation_postures",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dominant_main_code": {"type": "string", "enum": MAIN_CODES},
            "dominant_sub_code": {
                "type": "string",
                "description": "Must be a valid subcode for dominant_main_code; empty string if dominant_main_code is Uncoded.",
            },
            "all_codes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "main_code": {"type": "string", "enum": MAIN_CODES},
                        "sub_code": {"type": "string"},
                    },
                    "required": ["main_code", "sub_code"],
                },
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dominant": {"type": "string"},
                    "secondary": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 2,
                    },
                },
                "required": ["dominant", "secondary"],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {"type": "string"},
            "was_precoded": {"type": "boolean"},
        },
        "required": [
            "dominant_main_code",
            "dominant_sub_code",
            "all_codes",
            "evidence",
            "confidence",
            "notes",
            "was_precoded",
        ],
    },
}


# =========================================================
# 2) IO helpers
# =========================================================
def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}")


def write_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def guess_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "text", "content", "comment", "message",
        "评论", "弹幕", "danmu", "正文", "内容", "留言", "评论内容", "弹幕内容"
    ]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    best_col, best_score = None, -1
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).fillna("")
            score = s.str.len().mean()
            if score > best_score:
                best_score, best_col = score, c
    if best_col is None:
        raise ValueError("Could not guess text column. Please specify --text-col.")
    return best_col


def compact_text(x: str, max_len: int = 600) -> str:
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x[:max_len]


def load_api_key(path: str = "api_key.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("API key file is empty.")
    return key


def infer_platform_from_filename(fp: str) -> Optional[str]:
    name = os.path.basename(fp)
    if any(k in name for k in ["抖音", "douyin", "DY", "Douyin"]):
        return "Douyin"
    if any(k in name for k in ["小红书", "xiaohongshu", "XHS", "RED"]):
        return "Xiaohongshu"
    if any(k in name for k in ["B站", "哔哩哔哩", "bilibili", "Bilibili"]):
        return "Bilibili"
    return None


# =========================================================
# 3) Deterministic pre-coding (ultra-short formulaic items)
# =========================================================
EMOJI_BRACKET = re.compile(r"(\[[^\]]{1,6}\]){1,8}$")
EMOJI_UNICODE = re.compile(r"[\U00010000-\U0010ffff]")

PHATIC_EXACT = {
    "哈哈", "哈哈哈", "hhh", "笑死", "来了", "路过", "打卡", "已存", "收藏", "收藏了",
    "拿图", "借图", "礼貌拿图", "关注了", "求bgm", "bgm", "在哪", "在哪里", "谢谢", "谢谢你",
    "我也是", "是的", "确实", "懂了", "好家伙", "牛", "绝", "哇", "？", "?", "！", "!", "…", "。"
}

RITUAL_SHORTHAND = {
    "接", "接好运", "接福气", "接上岸", "上岸", "逢考必过", "保佑", "许愿", "还愿", "祈福",
    "泰山奶奶", "奶奶", "谢谢奶奶", "奶奶好", "碧霞", "碧霞祠", "接1111", "1111", "111"
}

RITUAL_PILGRIMAGE_CUES = {"朝圣", "祈福", "参拜", "进香", "还愿", "许愿"}
RITUAL_NIGHT_CUES = {"夜爬", "通宵", "凌晨", "天没亮"}
RITUAL_SUNRISE_CUES = {"日出", "看日出", "等日出", "到顶", "登顶"}
RITUAL_ENDURANCE_CUES = {"累", "腿废", "膝盖", "坚持", "走不动", "崩溃"}
RITUAL_BLESSING_CUES = {"接", "保佑", "逢考必过", "上岸", "接好运", "接福气", "接上岸"}

def _is_mostly_emoji_or_punct(t: str) -> bool:
    if not t:
        return True
    if EMOJI_BRACKET.fullmatch(t):
        return True
    cleaned = EMOJI_UNICODE.sub("", t)
    cleaned = re.sub(r"[!！?？。，…,.]+", "", cleaned).strip()
    return cleaned == ""

def pre_code(text: str) -> Optional[Dict[str, Any]]:
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return {
            "dominant_main_code": "Uncoded",
            "dominant_sub_code": "",
            "all_codes": [{"main_code": "Uncoded", "sub_code": ""}],
            "evidence": {"dominant": "", "secondary": []},
            "confidence": "low",
            "notes": "pre_code: empty text -> Uncoded",
            "was_precoded": True,
        }

    # Ritual shorthand
    if (
        t in RITUAL_SHORTHAND
        or t.startswith("接")
        or ("上岸" in t and len(t) <= 10)
        or ("1111" in t and len(t) <= 12)
        or any(x in t for x in ["泰山奶奶", "碧霞祠"])
    ):
        sub = "Vernacular Blessing"
        if any(c in t for c in RITUAL_NIGHT_CUES):
            sub = "Night Climb & Threshold"
        elif any(c in t for c in RITUAL_SUNRISE_CUES):
            sub = "Sunrise Completion"
        elif any(c in t for c in RITUAL_PILGRIMAGE_CUES):
            sub = "Pilgrimage Language"
        elif any(c in t for c in RITUAL_ENDURANCE_CUES):
            sub = "Endurance & Body Test"
        elif any(c in t for c in RITUAL_BLESSING_CUES):
            sub = "Vernacular Blessing"

        return {
            "dominant_main_code": "Ritualized Enactment",
            "dominant_sub_code": sub,
            "all_codes": [{"main_code": "Ritualized Enactment", "sub_code": sub}],
            "evidence": {"dominant": t, "secondary": []},
            "confidence": "high",
            "notes": "pre_code: ritual shorthand/formula",
            "was_precoded": True,
        }

    # Emoji-only / phatic / ultra short
    if _is_mostly_emoji_or_punct(t) or t in PHATIC_EXACT or len(t) <= 3:
        if _is_mostly_emoji_or_punct(t) or len(t) <= 2:
            sub = "Minimal Response"
        elif any(k in t for k in ["收藏", "已存", "拿图", "借图", "求bgm", "谢谢", "关注"]):
            sub = "Interaction Maintenance"
        else:
            sub = "Crowd Presence" if any(k in t for k in ["来了", "路过", "打卡"]) else "Minimal Response"

        return {
            "dominant_main_code": "Phatic / Ambient Participation",
            "dominant_sub_code": sub,
            "all_codes": [{"main_code": "Phatic / Ambient Participation", "sub_code": sub}],
            "evidence": {"dominant": t, "secondary": []},
            "confidence": "medium",
            "notes": "pre_code: phatic/etiquette/ultra-short",
            "was_precoded": True,
        }

    return None


# =========================================================
# 4) Boundary-case heuristics (for audit + goldset sampling)
#    Two zones you specified:
#    (i) ritual shorthand co-occurs with affective evaluation
#    (ii) irony/meme-like stance overlaps with negative affect
# =========================================================
RITUAL_CUES = {
    "接", "保佑", "许愿", "还愿", "祈福", "朝圣", "泰山奶奶", "碧霞祠", "封禅", "岱庙",
    "夜爬", "通宵", "日出", "登顶", "到顶"
}
AFFECT_CUES = {
    "震撼", "神圣", "壮观", "治愈", "放松", "感动", "想哭", "哭了", "泪目",
    "累", "腿废", "膝盖", "崩溃", "值", "不值", "后悔", "成功", "做到", "成就感"
}
IRONY_CUES = {
    "无语", "离谱", "笑死", "梗", "特种兵", "没苦硬吃", "营销", "作秀", "人比山多", "人多", "排队",
    "我真服了", "太抽象", "太逆天"
}
NEG_AFFECT_CUES = {
    "劝退", "不建议", "后悔", "不值", "坑", "受不了", "太累", "累死", "烦", "挤", "排到崩溃", "要命"
}

def _has_any(text: str, cues: set) -> bool:
    return any(c in text for c in cues)

def boundary_flags(text: str) -> Dict[str, Any]:
    t = str(text)
    ritual = _has_any(t, RITUAL_CUES)
    affect = _has_any(t, AFFECT_CUES)
    irony = _has_any(t, IRONY_CUES)
    neg = _has_any(t, NEG_AFFECT_CUES)

    zone_ritual_vs_affect = ritual and affect
    zone_irony_vs_neg = irony and neg

    # store brief matched cues for auditability (keep short)
    matched = []
    if zone_ritual_vs_affect:
        r = [c for c in RITUAL_CUES if c in t][:3]
        a = [c for c in AFFECT_CUES if c in t][:3]
        matched.append(f"ritual_vs_affect: ritual={r}, affect={a}")
    if zone_irony_vs_neg:
        i = [c for c in IRONY_CUES if c in t][:3]
        n = [c for c in NEG_AFFECT_CUES if c in t][:3]
        matched.append(f"irony_vs_neg: irony={i}, neg={n}")

    return {
        "boundary_ritual_vs_affect": int(zone_ritual_vs_affect),
        "boundary_irony_vs_neg": int(zone_irony_vs_neg),
        "boundary_any": int(zone_ritual_vs_affect or zone_irony_vs_neg),
        "boundary_notes": " | ".join(matched)[:240],
    }


# =========================================================
# 5) API coding
# =========================================================
class TransientAPIError(Exception):
    pass

def _is_insufficient_quota(err: Exception) -> bool:
    s = str(err).lower()
    return ("insufficient_quota" in s) or ("exceeded your current quota" in s)

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(TransientAPIError),
)
def code_one(client: OpenAI, model: str, text: str, platform_hint: Optional[str] = None) -> Dict[str, Any]:
    user_msg = ""
    if platform_hint:
        user_msg += f"Platform hint: {platform_hint}\n"
    user_msg += f"Text: {text}"

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CODEBOOK_TEXT},
                {"role": "user", "content": user_msg},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": CODING_SCHEMA["name"],
                    "strict": CODING_SCHEMA["strict"],
                    "schema": CODING_SCHEMA["schema"],
                }
            },
        )
        out_text = getattr(resp, "output_text", None)
        if not out_text:
            out_text = resp.output[0].content[0].text
        return json.loads(out_text)

    except Exception as e:
        # fail fast on insufficient quota
        if isinstance(e, RateLimitError) and _is_insufficient_quota(e):
            raise

        if isinstance(e, (APITimeoutError, APIConnectionError, APIError, RateLimitError)):
            raise TransientAPIError(str(e))

        msg = str(e).lower()
        # fallback if json_schema unsupported by chosen model endpoint
        if any(k in msg for k in ["json_schema", "unsupported", "invalid", "not supported"]):
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": CODEBOOK_TEXT + "\n\nReturn ONLY valid JSON."},
                        {"role": "user", "content": user_msg},
                    ],
                    text={"format": {"type": "json_object"}},
                )
                out_text = getattr(resp, "output_text", None)
                if not out_text:
                    out_text = resp.output[0].content[0].text
                return json.loads(out_text)
            except Exception as e2:
                if isinstance(e2, RateLimitError) and _is_insufficient_quota(e2):
                    raise
                if isinstance(e2, (APITimeoutError, APIConnectionError, APIError, RateLimitError)):
                    raise TransientAPIError(str(e2))
                raise

        if any(k in msg for k in ["timeout", "timed out", "temporarily", "503", "502", "server error"]):
            raise TransientAPIError(str(e))

        raise


# =========================================================
# 6) Summaries (dominant-code & any-code)
# =========================================================
def parse_all_codes(cell: str) -> List[Tuple[str, str]]:
    if not cell or not isinstance(cell, str):
        return []
    try:
        arr = json.loads(cell)
        out = []
        for obj in arr:
            m = obj.get("main_code", "")
            s = obj.get("sub_code", "")
            if m:
                out.append((m, s))
        return out
    except Exception:
        return []

def make_summary_main(df: pd.DataFrame, platform_label: str) -> pd.DataFrame:
    counts = df["dominant_main_code"].value_counts(dropna=False)
    total = int(counts.sum()) if len(counts) else 0
    rows = []
    for code in MAIN_CODES:
        n = int(counts.get(code, 0))
        rows.append({
            "platform": platform_label,
            "main_code": code,
            "n": n,
            "share_within_platform": (n / total) if total else 0.0,
        })
    return pd.DataFrame(rows)

def make_summary_anycode(df: pd.DataFrame, platform_label: str) -> pd.DataFrame:
    occ = {c: 0 for c in MAIN_CODES}
    total = len(df)
    for cell in df["all_codes_json"].tolist():
        pairs = parse_all_codes(cell)
        mains = set([m for (m, _) in pairs if m])
        for m in mains:
            if m in occ:
                occ[m] += 1
    rows = []
    for code in MAIN_CODES:
        n = int(occ.get(code, 0))
        rows.append({
            "platform": platform_label,
            "main_code": code,
            "n": n,
            "share_within_platform": (n / total) if total else 0.0,
        })
    return pd.DataFrame(rows)

def make_summary_subcode(df: pd.DataFrame, platform_label: str) -> pd.DataFrame:
    sub_counts = df.groupby(["dominant_main_code", "dominant_sub_code"]).size().reset_index(name="n")
    total = len(df)
    sub_counts["platform"] = platform_label
    sub_counts["share_within_platform"] = sub_counts["n"] / total if total else 0.0
    sub_counts = sub_counts[~((sub_counts["dominant_main_code"] == "Uncoded") & (sub_counts["dominant_sub_code"] != ""))]
    return sub_counts.sort_values(["dominant_main_code", "n"], ascending=[True, False])


# =========================================================
# 7) Gold-standard export (double-coding pack)
#    - stratify: random + low-confidence + boundary cases
#    - coders should NOT see model outputs
# =========================================================
def export_goldset(
    df_coded: pd.DataFrame,
    out_path: str,
    platform_hint: str,
    text_col: str,
    random_n: int = 200,
    lowconf_n: int = 0,
    boundary_n: int = 50,
    seed: int = 42,
) -> None:
    """
    Creates an Excel with two sheets:
    - coding_sheet: texts + blank coder columns (no model outputs, no boundary flags)
    - key_meta: row_id + boundary flags + (optional) model fields for researcher
    """

    df = df_coded.copy()
    df["row_id"] = df.index.astype(int)
    df["platform"] = platform_hint

    # boundary flags (precomputed preferred; if missing compute)
    needed = {"boundary_any", "boundary_ritual_vs_affect", "boundary_irony_vs_neg", "boundary_notes"}
    if not needed.issubset(set(df.columns)):
        flags = df["text_norm_600"].astype(str).apply(boundary_flags).apply(pd.Series)
        for c in flags.columns:
            df[c] = flags[c]

    # pools
    rng = seed
    pool_random = df.sample(n=min(random_n, len(df)), random_state=rng)

    pool_low = pd.DataFrame(columns=df.columns)
    if lowconf_n > 0 and "llm_confidence" in df.columns:
        low_pool = df[(df["llm_confidence"] == "low") & (df.get("was_precoded", False) == False)]
        if len(low_pool) > 0:
            pool_low = low_pool.sample(n=min(lowconf_n, len(low_pool)), random_state=rng + 7)

    pool_boundary = pd.DataFrame(columns=df.columns)
    boundary_pool = df[df["boundary_any"] == 1]
    if len(boundary_pool) > 0 and boundary_n > 0:
        pool_boundary = boundary_pool.sample(n=min(boundary_n, len(boundary_pool)), random_state=rng + 13)

    sel = pd.concat([pool_random, pool_low, pool_boundary], axis=0).drop_duplicates(subset=["row_id"])
    sel = sel.sample(frac=1.0, random_state=rng + 99)  # shuffle
    sel = sel.reset_index(drop=True)

    # sheet for coders (NO model outputs, NO boundary info)
    coding_sheet = pd.DataFrame({
        "case_id": [f"{platform_hint[:3]}_{i+1:04d}" for i in range(len(sel))],
        "row_id": sel["row_id"],
        "platform": sel["platform"],
        "text": sel["text_norm_600"],
        "coder1_main": "",
        "coder1_sub": "",
        "coder2_main": "",
        "coder2_sub": "",
        "adjudicated_main": "",
        "adjudicated_sub": "",
        "notes_for_adjudication": "",
    })

    # researcher key/meta (selection rationale + model fields)
    meta_cols = [
        "row_id", "platform", "text_norm_600",
        "boundary_any", "boundary_ritual_vs_affect", "boundary_irony_vs_neg", "boundary_notes",
    ]
    # include model fields if present (for researcher, not coders)
    for c in ["dominant_main_code", "dominant_sub_code", "llm_confidence", "was_precoded", "llm_notes"]:
        if c in sel.columns:
            meta_cols.append(c)
    key_meta = sel[meta_cols].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        coding_sheet.to_excel(w, index=False, sheet_name="coding_sheet")
        key_meta.to_excel(w, index=False, sheet_name="key_meta")


# =========================================================
# 8) Stability check (test–retest on a subset)
# =========================================================
def stability_check(client: OpenAI, model: str, df: pd.DataFrame, text_col: str, platform_hint: str,
                    sample_n: int = 1000, seed: int = 42, sleep: float = 0.0, no_precoding: bool = False) -> pd.DataFrame:
    subset = df.sample(n=min(sample_n, len(df)), random_state=seed).copy()
    subset["text_norm"] = subset[text_col].astype(str).apply(lambda x: compact_text(x, 600))

    def run_once(tag: str) -> List[str]:
        outs = []
        for t in tqdm(subset["text_norm"].tolist(), desc=f"Stability run {tag}"):
            if not no_precoding:
                pre = pre_code(t)
                if pre is not None:
                    outs.append(pre["dominant_main_code"])
                    continue
            data = code_one(client, model, t, platform_hint=platform_hint)
            outs.append(data.get("dominant_main_code", "Uncoded"))
            if sleep > 0:
                time.sleep(sleep)
        return outs

    r1 = run_once("A")
    r2 = run_once("B")

    subset["dominant_run_A"] = r1
    subset["dominant_run_B"] = r2
    subset["agree"] = (subset["dominant_run_A"] == subset["dominant_run_B"]).astype(int)
    agree_rate = subset["agree"].mean() if len(subset) else 0.0

    try:
        import krippendorff

        labels = sorted(set(r1) | set(r2))
        label_to_int = {label: i for i, label in enumerate(labels)}
        reliability_data = [
            [label_to_int[x] for x in r1],
            [label_to_int[x] for x in r2],
        ]
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="nominal"
        )
    except Exception:
        alpha = None

    return pd.DataFrame([{
        "platform_hint": platform_hint,
        "sample_n": len(subset),
        "run_to_run_agreement": float(agree_rate),
        "krippendorff_alpha": None if alpha is None else float(alpha),
    }])


# =========================================================
# 9) Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["抖音评论40.xlsx", "小红书评论40.xlsx", "B站弹幕20.csv"],
        help="Input files (.xlsx/.csv)"
    )
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--platform-col", default=None)
    ap.add_argument("--limit", type=int, default=0)  # 0 = all
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--outdir", default="coded_outputs")
    ap.add_argument("--no-precoding", action="store_true", help="Disable deterministic pre_code shortcut")
    ap.add_argument("--github-url", default="", help="Optional GitHub repo URL to store in outputs/metadata")

    # stability
    ap.add_argument("--stability-check", action="store_true", help="Run test–retest stability check on each file")
    ap.add_argument("--stability-n", type=int, default=1000, help="Sample size for stability check per file")

    # NEW: boundary export (audit candidates)
    ap.add_argument("--export-boundary", action="store_true", help="Export boundary-case candidates list")
    ap.add_argument("--boundary-max", type=int, default=2000, help="Max boundary candidates to export (0=all)")

    # NEW: goldset export (double coding pack)
    ap.add_argument("--export-goldset", action="store_true", help="Export a double-coding gold-standard pack (Excel)")
    ap.add_argument("--goldset-random-n", type=int, default=200, help="Random items per platform/file")
    ap.add_argument("--goldset-boundary-n", type=int, default=50, help="Boundary cases per platform/file")
    ap.add_argument("--goldset-lowconf-n", type=int, default=0,
                    help="Optional: extra low-confidence items (selection uses model but coders won't see it)")
    ap.add_argument("--goldset-seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    client = OpenAI(api_key=load_api_key(), timeout=120.0)

    # connectivity test
    try:
        _ = client.responses.create(model=args.model, input="ping")
    except Exception as e:
        print("API connectivity test failed:", repr(e))
        raise

    for in_path in args.inputs:
        df = read_any(in_path)

        text_col = args.text_col or guess_text_column(df)
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in {in_path}")

        platform_hint = None
        if args.platform_col and args.platform_col in df.columns:
            platform_hint = str(df[args.platform_col].dropna().iloc[0]) if df[args.platform_col].notna().any() else None
        if not platform_hint:
            platform_hint = infer_platform_from_filename(in_path) or "Unknown"

        n = len(df)
        if args.limit and args.limit > 0:
            n = min(n, args.limit)

        # normalize + boundary flags first (for audit & later goldset)
        df["text_norm_600"] = df[text_col].astype(str).apply(lambda x: compact_text(x, 600))
        flags = df["text_norm_600"].astype(str).apply(boundary_flags).apply(pd.Series)
        for c in flags.columns:
            df[c] = flags[c]

        dominant_main_list: List[str] = []
        dominant_sub_list: List[str] = []
        all_codes_json: List[str] = []
        evidence_json: List[str] = []
        conf_list: List[str] = []
        notes_list: List[str] = []
        precoded_list: List[bool] = []

        for i in tqdm(range(n), desc=f"Coding {os.path.basename(in_path)}"):
            text = df.at[i, "text_norm_600"]

            # deterministic pre-coding
            if not args.no_precoding:
                pre = pre_code(text)
                if pre is not None:
                    dominant_main_list.append(pre["dominant_main_code"])
                    dominant_sub_list.append(pre["dominant_sub_code"])
                    all_codes_json.append(json.dumps(pre["all_codes"], ensure_ascii=False))
                    evidence_json.append(json.dumps(pre["evidence"], ensure_ascii=False))
                    conf_list.append(pre["confidence"])
                    notes_list.append(pre["notes"])
                    precoded_list.append(True)
                    continue

            # LLM coding
            data = code_one(client, args.model, text, platform_hint=platform_hint)

            dominant_main_list.append(data.get("dominant_main_code", "Uncoded"))
            dominant_sub_list.append(data.get("dominant_sub_code", ""))
            all_codes_json.append(json.dumps(data.get("all_codes", []), ensure_ascii=False))
            evidence_json.append(json.dumps(data.get("evidence", {}), ensure_ascii=False))
            conf_list.append(data.get("confidence", "medium"))
            notes_list.append(data.get("notes", ""))
            precoded_list.append(bool(data.get("was_precoded", False)))

            if args.sleep > 0:
                time.sleep(args.sleep)

        # pad remaining rows if partially coded
        if n < len(df):
            pad = len(df) - n
            dominant_main_list += [""] * pad
            dominant_sub_list += [""] * pad
            all_codes_json += [""] * pad
            evidence_json += [""] * pad
            conf_list += [""] * pad
            notes_list += [""] * pad
            precoded_list += [False] * pad

        # attach fields
        df["dominant_main_code"] = dominant_main_list
        df["dominant_sub_code"] = dominant_sub_list
        df["all_codes_json"] = all_codes_json
        df["evidence_json"] = evidence_json
        df["llm_confidence"] = conf_list
        df["llm_notes"] = notes_list
        df["was_precoded"] = precoded_list
        df["llm_model"] = args.model
        df["platform_hint"] = platform_hint
        df["github_url"] = args.github_url

        base = os.path.splitext(os.path.basename(in_path))[0]

        # save coded file
        out_xlsx = os.path.join(args.outdir, f"{base}_coded.xlsx")
        write_any(df, out_xlsx)
        print(f"Saved coded file: {out_xlsx}")

        # save summaries (only coded part if limit)
        df_coded_part = df.iloc[:n].copy()
        sum_dom = make_summary_main(df_coded_part, platform_hint)
        sum_any = make_summary_anycode(df_coded_part, platform_hint)
        sum_sub = make_summary_subcode(df_coded_part, platform_hint)

        out_sum_dom = os.path.join(args.outdir, f"{base}_summary_maincode.csv")
        out_sum_any = os.path.join(args.outdir, f"{base}_summary_maincode_anycode.csv")
        out_sum_sub = os.path.join(args.outdir, f"{base}_summary_subcode.csv")

        sum_dom.to_csv(out_sum_dom, index=False)
        sum_any.to_csv(out_sum_any, index=False)
        sum_sub.to_csv(out_sum_sub, index=False)

        print(f"Saved summary (dominant): {out_sum_dom}")
        print(f"Saved summary (any-code): {out_sum_any}")
        print(f"Saved summary (subcode):  {out_sum_sub}")

        # NEW: export boundary-case candidates
        if args.export_boundary:
            bdf = df_coded_part[df_coded_part["boundary_any"] == 1].copy()
            if args.boundary_max and args.boundary_max > 0:
                bdf = bdf.head(args.boundary_max)
            out_b = os.path.join(args.outdir, f"{base}_boundary_candidates.csv")
            # keep it readable for audit
            keep_cols = [
                "platform_hint", "text_norm_600",
                "boundary_ritual_vs_affect", "boundary_irony_vs_neg", "boundary_notes",
                "dominant_main_code", "dominant_sub_code", "llm_confidence", "was_precoded"
            ]
            keep_cols = [c for c in keep_cols if c in bdf.columns]
            bdf[keep_cols].to_csv(out_b, index=False)
            print(f"Saved boundary candidates: {out_b}")

        # NEW: export goldset (double-coding pack)
        if args.export_goldset:
            out_g = os.path.join(args.outdir, f"{base}_goldset_double_coding.xlsx")
            export_goldset(
                df_coded=df_coded_part,
                out_path=out_g,
                platform_hint=platform_hint,
                text_col=text_col,
                random_n=args.goldset_random_n,
                lowconf_n=args.goldset_lowconf_n,
                boundary_n=args.goldset_boundary_n,
                seed=args.goldset_seed,
            )
            print(f"Saved goldset pack: {out_g}")

        # optional stability check
        if args.stability_check:
            rep = stability_check(
                client=client,
                model=args.model,
                df=df_coded_part.copy(),
                text_col="text_norm_600",
                platform_hint=platform_hint,
                sample_n=args.stability_n,
                seed=42,
                sleep=args.sleep,
                no_precoding=args.no_precoding,
            )
            out_stab = os.path.join(args.outdir, f"{base}_stability_report.csv")
            rep.to_csv(out_stab, index=False)
            print(f"Saved stability report: {out_stab}")


if __name__ == "__main__":
    main()