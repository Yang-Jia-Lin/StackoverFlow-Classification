# Scripts/04_test_nb.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import config
from Src.data.io import read_csv, validate_and_prepare, encode_labels
from Src.features.vectorize import load_sparse_matrix, load_vectorizer, transform_texts
from Src.models.multinomialNB import load_nb_model, predict_nb
from Src.eval.metrics import (
    evaluate_binary_classifier,
    save_json,
    save_text,
    plot_and_save_confusion_matrix,
)


def ensure_output_dirs() -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _paths_for_cached_test_features() -> Tuple[Path, Path]:
    x_test_path = config.MODELS_DIR / "X_test.npz"
    y_test_path = config.MODELS_DIR / "y_test.csv"
    return x_test_path, y_test_path


def load_test_features_prefer_cache() -> Tuple[sparse.csr_matrix, np.ndarray, str]:
    """
    优先从 Results/Models 读取已缓存特征；
    若不存在则从 Data/Processed + vectorizer 在线生成。
    返回：X_test, y_test, source_flag
    """
    x_test_path, y_test_path = _paths_for_cached_test_features()

    if x_test_path.exists() and y_test_path.exists():
        X_test = load_sparse_matrix(x_test_path)

        y_df = pd.read_csv(y_test_path)
        if y_df.shape[1] == 1:
            y_test = y_df.iloc[:, 0].to_numpy(dtype=int)
        else:
            for col in ["y", "label", "Tags", config.LABEL_COL]:
                if col in y_df.columns:
                    y_test = y_df[col].to_numpy(dtype=int)
                    break
            else:
                y_test = y_df.iloc[:, 0].to_numpy(dtype=int)

        return X_test, y_test, "cache"

    # 回退：从 Processed 读文本 + 加载 vectorizer 做 transform
    test_df = read_csv(config.PROCESSED_TEST_FILE)
    test_df = validate_and_prepare(test_df, dropna=True)

    texts = test_df[config.TEXT_COL]
    y_test = encode_labels(test_df[config.LABEL_COL]).to_numpy(dtype=int)

    vectorizer = load_vectorizer(config.VECTORIZER_PATH)
    X_test = transform_texts(vectorizer, texts)

    return X_test, y_test, "on_the_fly"


def main() -> None:
    ensure_output_dirs()

    # 1) 载入测试特征
    X_test, y_test, feature_src = load_test_features_prefer_cache()

    # 2) 载入模型并预测
    model = load_nb_model(config.NB_MODEL_PATH)
    y_pred = predict_nb(model, X_test).astype(int)

    # 3) 评估
    # 二分类建议明确 classes 的顺序：这里按 [0,1]（python/html）
    labels = np.array([0, 1], dtype=int)
    target_names = [config.INV_LABEL_MAP[i] for i in labels.tolist()]

    out = evaluate_binary_classifier(
        y_true=y_test,
        y_pred=y_pred,
        average=config.METRICS_AVERAGE,
        pos_label=config.POS_LABEL_ID,
        zero_division=config.ZERO_DIVISION,
        cm_normalize=config.CONFUSION_MATRIX_NORMALIZE,
        labels=labels,
        target_names=target_names,
    )

    # 4) 写结果到 Results/Metrics
    # 附加运行信息
    out.metrics.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "feature_source": feature_src,
            "test_samples": int(X_test.shape[0]),
            "num_features": int(X_test.shape[1]),
            "model_path": str(config.NB_MODEL_PATH),
            "vectorizer_path": str(config.VECTORIZER_PATH),
        }
    )

    save_json(out.metrics, config.NB_METRICS_JSON)
    save_text(out.report, config.NB_REPORT_TXT)

    # 5) 可选：混淆矩阵图
    if config.SAVE_CONFUSION_MATRIX_FIG:
        plot_and_save_confusion_matrix(
            out.cm,
            out_path=config.CONFUSION_MATRIX_PNG,
            labels=target_names,
            title="Confusion Matrix" if config.CONFUSION_MATRIX_NORMALIZE is None
            else f"Confusion Matrix (normalize={config.CONFUSION_MATRIX_NORMALIZE})",
        )

    # 6) 控制台打印关键指标
    print("[OK] Test completed.")
    print(f"  Feature source: {feature_src}")
    print(f"  Precision: {out.metrics['precision']:.6f}")
    print(f"  Recall   : {out.metrics['recall']:.6f}")
    print(f"  F1       : {out.metrics['f1']:.6f}")
    print(f"  Saved metrics -> {config.NB_METRICS_JSON}")
    print(f"  Saved report  -> {config.NB_REPORT_TXT}")
    if config.SAVE_CONFUSION_MATRIX_FIG:
        print(f"  Saved CM fig  -> {config.CONFUSION_MATRIX_PNG}")


if __name__ == "__main__":
    main()
