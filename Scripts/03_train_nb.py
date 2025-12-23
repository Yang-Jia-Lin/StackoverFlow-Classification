# Scripts/03_train_nb.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import sparse

import config
from Src.data.io import read_csv, validate_and_prepare, encode_labels
from Src.features.vectorize import load_sparse_matrix, load_vectorizer, transform_texts
from Src.models.multinomialNB import train_nb, save_nb_model


def ensure_models_dir() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _paths_for_cached_features() -> Tuple[Path, Path]:
    """
    与 Scripts/02_vectorize.py 对齐的默认缓存文件名。
    """
    x_train_path = config.MODELS_DIR / "X_train.npz"
    y_train_path = config.MODELS_DIR / "y_train.csv"
    return x_train_path, y_train_path


def load_train_features_prefer_cache() -> Tuple[sparse.csr_matrix, np.ndarray, str]:
    """
    优先从 Results/Models 读取已缓存特征；
    若不存在则从 Data/Processed + vectorizer 在线生成。
    返回：X_train, y_train, source_flag
    """
    x_train_path, y_train_path = _paths_for_cached_features()

    if x_train_path.exists() and y_train_path.exists():
        X_train = load_sparse_matrix(x_train_path)

        # y_train.csv 由 02_vectorize.py 写出，通常是一列
        y_df = pd.read_csv(y_train_path)
        if y_df.shape[1] == 1:
            y_train = y_df.iloc[:, 0].to_numpy(dtype=int)
        else:
            # 如果包含多列，尝试取名为 y 或 label 的列，否则取第一列
            for col in ["y", "label", "Tags", config.LABEL_COL]:
                if col in y_df.columns:
                    y_train = y_df[col].to_numpy(dtype=int)
                    break
            else:
                y_train = y_df.iloc[:, 0].to_numpy(dtype=int)

        return X_train, y_train, "cache"

    # 回退：从 Processed 读文本 + 加载 vectorizer 做 transform
    train_df = read_csv(config.PROCESSED_TRAIN_FILE)
    train_df = validate_and_prepare(train_df, dropna=True)

    texts = train_df[config.TEXT_COL]
    y_train = encode_labels(train_df[config.LABEL_COL]).to_numpy(dtype=int)

    vectorizer = load_vectorizer(config.VECTORIZER_PATH)
    X_train = transform_texts(vectorizer, texts)

    return X_train, y_train, "on_the_fly"


def save_label_map(path: Path) -> None:
    """
    保存标签映射，便于后续推理/解释输出。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "label_map": config.LABEL_MAP,
        "inv_label_map": config.INV_LABEL_MAP,
        "pos_label_name": getattr(config, "POS_LABEL_NAME", None),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_models_dir()

    # 1) 加载训练特征
    X_train, y_train, src = load_train_features_prefer_cache()

    # 2) 训练 MultinomialNB
    out = train_nb(X_train, y_train, nb_params=config.NB_PARAMS)
    model = out.model

    # 3) 保存模型与标签映射
    save_nb_model(model, config.NB_MODEL_PATH)
    save_label_map(config.LABEL_MAP_PATH)

    # 4) 写训练运行日志（建议）
    log_lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"feature_source: {src}",
        f"train_samples: {X_train.shape[0]}",
        f"num_features: {X_train.shape[1]}",
        f"classes_: {out.classes_.tolist()}",
        f"nb_params: {config.NB_PARAMS}",
        f"model_path: {config.NB_MODEL_PATH}",
        f"label_map_path: {config.LABEL_MAP_PATH}",
    ]
    (config.MODELS_DIR / "train_nb_run.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("[OK] Training completed.")
    print(f"  Feature source: {src}")
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Num features : {X_train.shape[1]}")
    print(f"  Saved model  -> {config.NB_MODEL_PATH}")
    print(f"  Saved labels -> {config.LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
