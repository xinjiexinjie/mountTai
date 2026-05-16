import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report

GOLD_FILE = "goldset_adjudicated.xlsx"

df = pd.read_excel(GOLD_FILE, sheet_name="coding_sheet")

# Human inter-coder reliability
df_human = df.dropna(subset=["coder1_main", "coder2_main"])
df_human = df_human[
    (df_human["coder1_main"].astype(str).str.strip() != "") &
    (df_human["coder2_main"].astype(str).str.strip() != "")
]

kappa = cohen_kappa_score(df_human["coder1_main"], df_human["coder2_main"])
print("Cohen's kappa:", round(kappa, 3))

# Model-human agreement
# 需要 goldset 里有 llm_main 或者从 key_meta 合并 dominant_main_code
meta = pd.read_excel(GOLD_FILE, sheet_name="key_meta")
merged = df.merge(meta[["row_id", "dominant_main_code"]], on="row_id", how="left")

df_eval = merged.dropna(subset=["adjudicated_main", "dominant_main_code"])
df_eval = df_eval[
    (df_eval["adjudicated_main"].astype(str).str.strip() != "") &
    (df_eval["dominant_main_code"].astype(str).str.strip() != "")
]

y_true = df_eval["adjudicated_main"].astype(str).str.strip()
y_pred = df_eval["dominant_main_code"].astype(str).str.strip()

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print("Model-human accuracy:", round(acc, 3))
print("Model-human macro-F1:", round(macro_f1, 3))
print(classification_report(y_true, y_pred))