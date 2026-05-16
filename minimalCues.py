import pandas as pd
import re
import unicodedata

IN_XLSX = "uncoded_context_validation_sample_pack_v2.xlsx"
OUT_XLSX = "uncoded_minimal_cues_autolabeled_sample_pack_v2.xlsx"

# ---- minimal-cues keyword lists (edit as you like) ----
CUES = {
    "Governance & Official Discourse": ["文旅","传承","正能量","高质量发展","文化强国","弘扬","官方","宣传","文明"],
    "Ritualized Enactment": ["接好运","接上岸","逢考必过","保佑","许愿","还愿","祈福","朝圣","进香","参拜",
                            "泰山奶奶","碧霞祠","岱庙","封禅","夜爬","通宵","凌晨","日出","登顶","到顶"],
    "Lifestyle Translation": ["攻略","路线","门票","索道","缆车","停车","交通","高铁","住宿","酒店","民宿",
                              "票价","价格","几点","时间","怎么去","怎么走","建议","装备","穿搭","出片","汉服","拍照","机位"],
    "Ironic Distancing": ["无语","离谱","笑死","人比山多","看人不看山","排队","人多","挤","劝退","没苦硬吃","特种兵","营销","作秀"],
    "Interpretive Framing": ["科普","讲解","你知道吗","历史","诗","典故","朝代","皇帝","封禅","岱庙"],
    "Affective Compression": ["震撼","神圣","壮观","治愈","放松","泪目","想哭","哭了","值了","值得","后悔","成就感","圆梦","累死","腿废","膝盖","崩溃"],
    "Phatic / Ambient Participation": ["已存","收藏","收藏了","拿图","借图","求bgm","bgm","谢谢","来了","路过","打卡","哈哈","哈哈哈","哇","牛","绝","懂了","我也是","在哪","在哪里"]
}
QUESTION_CUES = ["怎么","咋","在哪","在哪里","几点","多少","求","有没有","能不能","可以吗","为啥","为什么","啥","吗","？","?"]

def is_mostly_emoji_or_punct(t: str) -> bool:
    t = str(t).strip()
    if not t:
        return True
    s = re.sub(r"\s+", "", t)
    s = re.sub(r"[!！?？。，、,.…:：;；（）()\[\]{}<>《》\"'“”‘’\-—_~`]+", "", s)
    s = re.sub(r"\d+", "", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] not in ["S","P"])
    return s.strip() == ""

def classify_minimal_cues(text: str):
    t = str(text).strip()
    if not t:
        return False, "", "", "empty"
    if is_mostly_emoji_or_punct(t) or len(t) <= 2:
        return True, "Phatic / Ambient Participation", "Minimal Response", "emoji/punct or ultra-short"

    addressed = bool(re.search(r"(^@|回复|楼主|楼上|楼下)", t))
    is_question = any(q in t for q in QUESTION_CUES)

    hits = {}
    for code, kws in CUES.items():
        for kw in kws:
            if kw in t:
                hits.setdefault(code, []).append(kw)

    if addressed and not hits and not is_question:
        return False, "", "", "addressed-talk marker without posture cue"
    if not hits:
        if is_question and len(t) <= 12:
            return True, "Phatic / Ambient Participation", "Interaction Maintenance", "short question/request -> phatic"
        return False, "", "", "no strong minimal cues"

    # priority (conservative)
    priority = ["Governance & Official Discourse","Ritualized Enactment","Lifestyle Translation",
                "Ironic Distancing","Interpretive Framing","Affective Compression","Phatic / Ambient Participation"]
    chosen = next(p for p in priority if p in hits)

    # question + lifestyle -> lifestyle
    if is_question and "Lifestyle Translation" in hits:
        chosen = "Lifestyle Translation"

    # subcode rough mapping
    sub = ""
    if chosen == "Phatic / Ambient Participation":
        sub = "Interaction Maintenance" if any(k in t for k in ["收藏","已存","拿图","借图","求bgm","谢谢","在哪","在哪里"]) else "Minimal Response"
    elif chosen == "Ritualized Enactment":
        sub = "Night Climb & Threshold" if any(k in t for k in ["夜爬","通宵","凌晨"]) else "Vernacular Blessing"
    elif chosen == "Lifestyle Translation":
        sub = "Itinerary Optimization"
    elif chosen == "Ironic Distancing":
        sub = "Crowd Irony" if any(k in t for k in ["人多","排队","挤","人比山多"]) else "Meme Language"
    elif chosen == "Interpretive Framing":
        sub = "Educational Tone" if any(k in t for k in ["科普","讲解","你知道吗"]) else "Historical Reference"
    elif chosen == "Affective Compression":
        sub = "Awe & Sublimity"
    elif chosen == "Governance & Official Discourse":
        sub = "Institutional Voice"

    return True, chosen, sub, f"cues={ {k: hits[k][:3] for k in hits} }"

# ---- run on the sample pack ----
xl = pd.ExcelFile(IN_XLSX)
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        res = df["text"].apply(classify_minimal_cues)

        df["resolvable_with_minimal_cues"] = res.apply(lambda x: "Yes" if x[0] else "No")
        df["recode_main_with_minimal_cues"] = res.apply(lambda x: x[1] if x[0] else "")
        df["recode_sub_with_minimal_cues"] = res.apply(lambda x: x[2] if x[0] else "")
        df["minimal_cues_notes"] = res.apply(lambda x: x[3])

        df.to_excel(w, index=False, sheet_name=sheet)

print("Saved:", OUT_XLSX)