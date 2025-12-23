from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


INPUT_CSV = Path("/workspace/user/Coding/jialin/StackoverflowClassification/Data/Raw/StemmedData_30000.csv")
OUT_DIR = Path("/workspace/user/Coding/jialin/StackoverflowClassification/Data/five_fold")

N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42

# 列名（按你数据格式）
ID_COL = "Id"
TEXT_COL = "Title"
LABEL_COL = "Tags"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    for col in [ID_COL, TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found columns: {list(df.columns)}")

    # 统一标签格式（避免出现 'HTML'/'html' 混杂导致分层异常）
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    # 基础清理：去掉空文本/空标签
    df[TEXT_COL] = df[TEXT_COL].astype(str).replace({"nan": "", "None": ""})
    df = df[(df[TEXT_COL].str.strip() != "") & (df[LABEL_COL].str.strip() != "")].reset_index(drop=True)

    y = df[LABEL_COL].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

    summary_lines = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        fold_dir = OUT_DIR / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_path = fold_dir / f"fold_{fold_idx}_train.csv"
        val_path = fold_dir / f"fold_{fold_idx}_val.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")

        # 统计每折标签数量，便于你确认“均衡”
        train_counts = train_df[LABEL_COL].value_counts().to_dict()
        val_counts = val_df[LABEL_COL].value_counts().to_dict()

        summary_lines.append(
            f"fold_{fold_idx}: "
            f"train={len(train_df)} {train_counts} | "
            f"val={len(val_df)} {val_counts}"
        )

    summary_path = OUT_DIR / "five_fold_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[OK] Five-fold datasets generated.")
    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUT_DIR}")
    print(f"Summary written to: {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
