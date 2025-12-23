# Scripts/02_vectorize.py
from __future__ import annotations
from datetime import datetime
import config
from Src.data.io import read_csv, validate_and_prepare, encode_labels
from Src.features.vectorize import (
    build_vectorizer,
    fit_transform_texts,
    get_feature_names,
    save_vectorizer,
    save_sparse_matrix,
)


def ensure_out_dir() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_out_dir()

    # 1) 读取 Processed 的 train/test
    train_df = read_csv(config.PROCESSED_TRAIN_FILE)
    test_df = read_csv(config.PROCESSED_TEST_FILE)

    train_df = validate_and_prepare(train_df, dropna=True)
    test_df = validate_and_prepare(test_df, dropna=True)

    X_text_train = train_df[config.TEXT_COL]
    y_train = encode_labels(train_df[config.LABEL_COL])

    X_text_test = test_df[config.TEXT_COL]
    y_test = encode_labels(test_df[config.LABEL_COL])

    # 2) 构建并 fit/transform
    vectorizer = build_vectorizer(
        vectorizer_type=getattr(config, "VECTORIZER_TYPE", "count"),
        vectorizer_params=config.VECTORIZER_PARAMS,
    )
    vectorizer, X_train, X_test = fit_transform_texts(X_text_train, X_text_test, vectorizer=vectorizer)

    # 3) 保存向量器与稀疏矩阵（npz）
    save_vectorizer(vectorizer, config.VECTORIZER_PATH)

    x_train_path = config.MODELS_DIR / "X_train.npz"
    x_test_path = config.MODELS_DIR / "X_test.npz"
    y_train_path = config.MODELS_DIR / "y_train.csv"
    y_test_path = config.MODELS_DIR / "y_test.csv"

    save_sparse_matrix(X_train, x_train_path)
    if X_test is not None:
        save_sparse_matrix(X_test, x_test_path)

    # 标签也保存一下，后续 train/test 脚本可直接读取（可选但很方便）
    y_train.to_csv(y_train_path, index=False, header=True)
    y_test.to_csv(y_test_path, index=False, header=True)

    # 4) 保存词表（可选）
    vocab = get_feature_names(vectorizer)
    vocab_path = config.MODELS_DIR / "vocab.txt"
    vocab_path.write_text("\n".join(vocab.tolist()), encoding="utf-8")

    # 5) 写运行日志（可选）
    info = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"train_rows: {len(train_df)}",
        f"test_rows: {len(test_df)}",
        f"vocab_size: {len(vocab)}",
        f"vectorizer_type: {getattr(config, 'VECTORIZER_TYPE', 'count')}",
        f"vectorizer_params: {config.VECTORIZER_PARAMS}",
        f"vectorizer_path: {config.VECTORIZER_PATH}",
        f"X_train_path: {x_train_path}",
        f"X_test_path: {x_test_path}",
        f"y_train_path: {y_train_path}",
        f"y_test_path: {y_test_path}",
        f"vocab_path: {vocab_path}",
    ]
    (config.MODELS_DIR / "vectorize_run.txt").write_text("\n".join(info), encoding="utf-8")

    print("[OK] Vectorization completed.")
    print(f"  Saved vectorizer -> {config.VECTORIZER_PATH}")
    print(f"  Saved X_train -> {x_train_path}")
    print(f"  Saved X_test  -> {x_test_path}")
    print(f"  Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
