# Scripts/01_preprocess.py
from __future__ import annotations
from pathlib import Path

import config
from Src.data.io import read_csv, validate_and_prepare, save_csv
from Src.data.preprocess import preprocess_dataframe


def ensure_processed_dir() -> None:
    """确保 Data/Processed 目录存在。"""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_file(input_path: Path, output_path: Path, *, do_stemming: bool = False) -> None:
    """
    读取 -> 校验 -> 预处理 -> 保存
    """
    df = read_csv(input_path)
    df = validate_and_prepare(df, dropna=True)

    df_out = preprocess_dataframe(
        df,
        text_col=config.TEXT_COL,
        output_col=None,  # 覆盖原列 Title_Stemmed
        do_lowercase=True,
        remove_punct=True,
        do_squeeze_whitespace=True,
        do_stemming=do_stemming,
        stemmer_type="porter",
    )

    save_csv(df_out, output_path, index=False)

    # 简单日志
    print(f"[OK] {input_path.name} -> {output_path.name} | rows={len(df_out)}")
    print(df_out[[c for c in [config.ID_COL, config.TEXT_COL, config.LABEL_COL] if c in df_out.columns]].head(3))
    print("-" * 60)


def main() -> None:
    ensure_processed_dir()

    # 重要：你当前 Title_Stemmed 已经词干化，默认不再二次词干化
    DO_STEMMING = False

    preprocess_file(config.RAW_TRAIN_FILE, config.PROCESSED_TRAIN_FILE, do_stemming=DO_STEMMING)
    preprocess_file(config.RAW_TEST_FILE, config.PROCESSED_TEST_FILE, do_stemming=DO_STEMMING)

    print(f"[DONE] Preprocessed files are saved under: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
