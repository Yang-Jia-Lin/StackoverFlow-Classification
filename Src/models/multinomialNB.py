# Src/models/multinomialNB.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

import config


MatrixLike = Union[sparse.spmatrix, np.ndarray]


@dataclass
class NBTrainOutput:
    model: MultinomialNB
    classes_: np.ndarray


def build_nb_model(nb_params: Optional[Dict[str, Any]] = None) -> MultinomialNB:
    """
    构建 MultinomialNB 模型。
    nb_params: 若不传则使用 config.NB_PARAMS
    """
    params = dict(nb_params) if nb_params is not None else dict(config.NB_PARAMS)
    return MultinomialNB(**params)


def train_nb(
    X_train: MatrixLike,
    y_train: np.ndarray,
    *,
    nb_params: Optional[Dict[str, Any]] = None
) -> NBTrainOutput:
    """
    训练 MultinomialNB。
    """
    model = build_nb_model(nb_params=nb_params)
    model.fit(X_train, y_train)
    return NBTrainOutput(model=model, classes_=model.classes_)


def predict_nb(model: MultinomialNB, X: MatrixLike) -> np.ndarray:
    """预测类别（0/1）。"""
    return model.predict(X)


def predict_proba_nb(model: MultinomialNB, X: MatrixLike) -> np.ndarray:
    """预测概率。"""
    return model.predict_proba(X)


def save_nb_model(model: MultinomialNB, path: Path) -> None:
    """保存模型（joblib）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_nb_model(path: Path) -> MultinomialNB:
    """加载模型（joblib）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NB model not found: {path}")
    return joblib.load(path)
