from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import config
from Src.data.io import read_csv, validate_and_prepare, encode_labels
from Src.features.vectorize import build_vectorizer, fit_transform_texts, get_feature_names, save_vectorizer
from Src.models.multinomialNB import train_nb, predict_nb, predict_proba_nb, save_nb_model
from Src.eval.metrics import (
    evaluate_binary_classifier,
    save_json,
    save_text,
    plot_and_save_confusion_matrix,
)


# -----------------------------
# 配置：五折数据位置
# -----------------------------
FIVE_FOLD_DIR = config.PROCESSED_DIR / "five_fold"  # Data/Processed/five_fold


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir() -> Path:
    run_dir = config.RESULTS_DIR / f"five_fold_cv_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "folds").mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_fold_files(fold_dir: Path, fold_idx: int) -> Tuple[Path, Path]:
    """
    兼容多种命名：
    - fold_{i}_train.csv + fold_{i}_val.csv
    - fold_{i}_train.csv + fold_{i}_test.csv
    - train.csv + val.csv (或 test.csv)
    """
    candidates_train = [
        fold_dir / f"fold_{fold_idx}_train.csv",
        fold_dir / "train.csv",
        fold_dir / f"train_fold_{fold_idx}.csv",
    ]
    candidates_val = [
        fold_dir / f"fold_{fold_idx}_val.csv",
        fold_dir / f"fold_{fold_idx}_test.csv",
        fold_dir / "val.csv",
        fold_dir / "test.csv",
        fold_dir / f"val_fold_{fold_idx}.csv",
        fold_dir / f"test_fold_{fold_idx}.csv",
    ]

    train_path = next((p for p in candidates_train if p.exists()), None)
    val_path = next((p for p in candidates_val if p.exists()), None)

    if train_path is None or val_path is None:
        raise FileNotFoundError(
            f"Cannot find train/val files in {fold_dir}. "
            f"Expected something like fold_{fold_idx}_train.csv and fold_{fold_idx}_val.csv."
        )
    return train_path, val_path


def _plot_and_save_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> Dict[str, float]:
    """
    y_score: 正类（POS_LABEL_ID）的预测概率
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=config.POS_LABEL_ID)
    roc_auc = auc(fpr, tpr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"roc_auc": float(roc_auc)}


def _load_fold_df(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    df = validate_and_prepare(df, dropna=True)  # 会校验列名/标签合法性等
    return df


def run_one_fold(fold_idx: int, fold_dir: Path, run_dir: Path) -> Dict[str, float]:
    """
    单折：向量化 -> 训练 -> 评估 -> 输出文件
    """
    train_path, val_path = _find_fold_files(fold_dir, fold_idx)

    df_train = _load_fold_df(train_path)
    df_val = _load_fold_df(val_path)

    x_text_train = df_train[config.TEXT_COL]
    y_train = encode_labels(df_train[config.LABEL_COL]).to_numpy(dtype=int)

    x_text_val = df_val[config.TEXT_COL]
    y_val = encode_labels(df_val[config.LABEL_COL]).to_numpy(dtype=int)

    # 1) 向量化（每折重新 fit，避免泄漏）
    vectorizer = build_vectorizer(
        vectorizer_type=getattr(config, "VECTORIZER_TYPE", "count"),
        vectorizer_params=config.VECTORIZER_PARAMS,
    )
    vectorizer, X_train, X_val = fit_transform_texts(x_text_train, x_text_val, vectorizer=vectorizer)

    # 2) 训练 NB
    nb_out = train_nb(X_train, y_train, nb_params=config.NB_PARAMS)
    model = nb_out.model

    # 3) 预测 + 概率（用于 ROC）
    y_pred = predict_nb(model, X_val).astype(int)
    proba = predict_proba_nb(model, X_val)  # shape: [n, n_classes]
    # 找到正类（html=1）对应的概率列
    classes = model.classes_.tolist()
    if config.POS_LABEL_ID not in classes:
        raise RuntimeError(f"POS_LABEL_ID={config.POS_LABEL_ID} not in model.classes_={classes}")
    pos_col = classes.index(config.POS_LABEL_ID)
    y_score = proba[:, pos_col]

    # 4) 计算 P/R/F1/Acc + CM + report
    labels = np.array([0, 1], dtype=int)
    target_names = [config.INV_LABEL_MAP[i] for i in labels.tolist()]

    eval_out = evaluate_binary_classifier(
        y_true=y_val,
        y_pred=y_pred,
        average=config.METRICS_AVERAGE,
        pos_label=config.POS_LABEL_ID,
        zero_division=config.ZERO_DIVISION,
        cm_normalize=config.CONFUSION_MATRIX_NORMALIZE,
        labels=labels,
        target_names=target_names,
    )

    # 5) ROC-AUC
    roc_auc = float(roc_auc_score(y_val, y_score))

    # 6) 输出文件
    fold_out_dir = run_dir / "folds" / f"fold_{fold_idx}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型与向量器（每折一份，便于复现）
    save_vectorizer(vectorizer, fold_out_dir / "vectorizer.joblib")
    save_nb_model(model, fold_out_dir / "nb_model.joblib")

    # 保存词表（可选，调试用）
    vocab = get_feature_names(vectorizer)
    (fold_out_dir / "vocab.txt").write_text("\n".join(vocab.tolist()), encoding="utf-8")

    # 保存指标与报告
    metrics = dict(eval_out.metrics)
    metrics.update(
        {
            "fold": fold_idx,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "num_features": int(X_train.shape[1]),
            "roc_auc": roc_auc,
            "nb_params": config.NB_PARAMS,
            "vectorizer_params": config.VECTORIZER_PARAMS,
        }
    )
    save_json(metrics, fold_out_dir / "metrics.json")
    save_text(eval_out.report, fold_out_dir / "classification_report.txt")

    # 混淆矩阵图（可选）
    if config.SAVE_CONFUSION_MATRIX_FIG:
        plot_and_save_confusion_matrix(
            eval_out.cm,
            out_path=fold_out_dir / "confusion_matrix.png",
            labels=target_names,
            title="Confusion Matrix"
            if config.CONFUSION_MATRIX_NORMALIZE is None
            else f"Confusion Matrix (normalize={config.CONFUSION_MATRIX_NORMALIZE})",
        )

    # ROC 曲线图
    _plot_and_save_roc(
        y_true=y_val,
        y_score=y_score,
        out_path=fold_out_dir / "roc_curve.png",
        title=f"ROC Curve (fold {fold_idx})",
    )

    # 返回用于汇总的关键数值
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
    }


def main() -> None:
    if not FIVE_FOLD_DIR.exists():
        raise FileNotFoundError(f"Five-fold dir not found: {FIVE_FOLD_DIR}")

    run_dir = _make_run_dir()

    # 默认按 fold_1 ... fold_5 子目录
    fold_dirs = []
    for i in range(1, 6):
        d = FIVE_FOLD_DIR / f"fold_{i}"
        if d.exists():
            fold_dirs.append((i, d))

    if len(fold_dirs) != 5:
        # 若不是标准命名，则尝试扫描 five_fold 下所有 fold_* 目录
        auto = sorted([p for p in FIVE_FOLD_DIR.glob("fold_*") if p.is_dir()])
        if len(auto) >= 5:
            fold_dirs = [(idx + 1, auto[idx]) for idx in range(5)]
        else:
            raise RuntimeError(
                f"Expected 5 fold dirs under {FIVE_FOLD_DIR}, got {len(fold_dirs)}. "
                "Please ensure structure like five_fold/fold_1 ... fold_5."
            )

    per_fold: List[Dict[str, float]] = []
    for fold_idx, fold_dir in fold_dirs:
        print(f"[RUN] fold {fold_idx} -> {fold_dir}")
        stats = run_one_fold(fold_idx, fold_dir, run_dir)
        per_fold.append(stats)
        print(f"  acc={stats['accuracy']:.4f}  f1={stats['f1']:.4f}  auc={stats['roc_auc']:.4f}")

    # 汇总 mean ± std
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for k in keys:
        vals = np.array([d[k] for d in per_fold], dtype=float)
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std(ddof=1))  # 样本标准差
        summary[f"{k}_per_fold"] = [float(v) for v in vals.tolist()]

    save_json(summary, run_dir / "cv_summary.json")

    # 也写一个更易读的 txt
    lines = [f"Five-Fold CV Summary ({summary['timestamp']})", "-" * 60]
    for k in keys:
        lines.append(
            f"{k:10s}: mean={summary[f'{k}_mean']:.6f}  std={summary[f'{k}_std']:.6f}  "
            f"per_fold={summary[f'{k}_per_fold']}"
        )
    (run_dir / "cv_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] Five-fold CV finished.")
    print(f"Results saved to: {run_dir}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
