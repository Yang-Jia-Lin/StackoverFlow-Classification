# Src/features/vectorize.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config


VectorizerType = Union[CountVectorizer, TfidfVectorizer]


@dataclass
class VectorizationOutput:
    """向量化输出载体（可选，用于调试/记录）"""
    X_train: sparse.csr_matrix
    X_test: Optional[sparse.csr_matrix]
    feature_names: Optional[np.ndarray]


def build_vectorizer(
    *,
    vectorizer_type: str = "count",
    vectorizer_params: Optional[Dict[str, Any]] = None,
) -> VectorizerType:
    """
    构建向量器：
    - vectorizer_type: "count" 或 "tfidf"
    - vectorizer_params: 若不传则取 config.VECTORIZER_PARAMS
    """
    vtype = (vectorizer_type or "count").lower()
    params = dict(vectorizer_params) if vectorizer_params is not None else dict(config.VECTORIZER_PARAMS)

    if vtype == "count":
        return CountVectorizer(**params)
    if vtype == "tfidf":
        # TFIDF 需要组合参数：可复用 VECTORIZER_PARAMS 作为 tokenizer/过滤等配置
        tfidf_params = dict(getattr(config, "TFIDF_PARAMS", {}))
        # 允许外部传入覆盖
        tfidf_params.update(params)
        return TfidfVectorizer(**tfidf_params)

    raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}. Use 'count' or 'tfidf'.")


def fit_vectorizer(texts, *, vectorizer: Optional[VectorizerType] = None) -> VectorizerType:
    """
    在训练集上拟合向量器（只 fit，不 transform）。
    """
    if vectorizer is None:
        vectorizer = build_vectorizer(vectorizer_type=getattr(config, "VECTORIZER_TYPE", "count"))
    vectorizer.fit(texts)
    return vectorizer


def transform_texts(vectorizer: VectorizerType, texts) -> sparse.csr_matrix:
    """
    将文本转为稀疏矩阵。
    """
    X = vectorizer.transform(texts)
    # 确保 CSR（后续训练/保存最常用）
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    return X


def fit_transform_texts(
    train_texts,
    test_texts=None,
    *,
    vectorizer: Optional[VectorizerType] = None,
) -> Tuple[VectorizerType, sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """
    在训练集 fit，并 transform train/test。
    """
    if vectorizer is None:
        vectorizer = build_vectorizer(vectorizer_type=getattr(config, "VECTORIZER_TYPE", "count"))

    X_train = vectorizer.fit_transform(train_texts)
    if not sparse.isspmatrix_csr(X_train):
        X_train = X_train.tocsr()

    X_test = None
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        if not sparse.isspmatrix_csr(X_test):
            X_test = X_test.tocsr()

    return vectorizer, X_train, X_test


def get_feature_names(vectorizer: VectorizerType) -> np.ndarray:
    """
    获取词表（feature names）。
    兼容 sklearn 新旧接口。
    """
    if hasattr(vectorizer, "get_feature_names_out"):
        return vectorizer.get_feature_names_out()
    return np.array(vectorizer.get_feature_names())


def save_vectorizer(vectorizer: VectorizerType, path: Path) -> None:
    """
    保存向量器（joblib）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path) -> VectorizerType:
    """
    读取向量器（joblib）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {path}")
    return joblib.load(path)


def save_sparse_matrix(X: sparse.csr_matrix, path: Path) -> None:
    """
    保存稀疏矩阵为 .npz（推荐）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, X)


def load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    """
    读取 .npz 稀疏矩阵。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sparse matrix not found: {path}")
    X = sparse.load_npz(path)
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    return X
