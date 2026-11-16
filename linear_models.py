from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _to_feature_matrix(features: Any) -> np.ndarray:
    """Преобразует входные признаки в 2D массив float64."""
    if isinstance(features, pd.DataFrame):
        df = features.copy()
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "category":
                df[col] = df[col].map({"Yes": 1.0, "No": 0.0}).astype(float)
        arr = df.to_numpy(dtype=float)
    else:
        arr = np.asarray(features, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _to_target_vector(target: Any) -> np.ndarray:
    """Преобразует целевую переменную в 1D массив float64."""
    return np.asarray(target, dtype=float).reshape(-1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Численно стабильная сигмоида."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))


def _compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    """Вычисляет набор метрик бинарной классификации."""
    y_true_int = np.asarray(y_true, dtype=int).reshape(-1)
    preds_int = np.asarray(predictions, dtype=int).reshape(-1)
    probas_flt = np.asarray(probabilities, dtype=float).reshape(-1)

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true_int, probas_flt)),
        accuracy=float(accuracy_score(y_true_int, preds_int)),
        precision=float(precision_score(y_true_int, preds_int, zero_division=0)),
        recall=float(recall_score(y_true_int, preds_int, zero_division=0)),
        f1=float(f1_score(y_true_int, preds_int, zero_division=0)),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = _to_feature_matrix(data.train_features)
    y_train = _to_target_vector(data.train_target)
    X_test = _to_feature_matrix(data.test_features)

    n_samples, n_features = X_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)

    needs_internal_scaling = bool(np.max(np.abs(X_train)) > 15.0)
    if needs_internal_scaling:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        X_train_opt = (X_train - mean) / std
    else:
        X_train_opt = X_train

    weights_opt = np.zeros(n_features, dtype=float)
    bias_opt = 0.0
    loss_history: list[float] = []

    for _ in range(epochs):
        y_pred_train = X_train_opt @ weights_opt + bias_opt
        error = y_pred_train - y_train

        mse_loss = float(np.mean(error ** 2))
        loss_history.append(mse_loss)

        grad_w = (2.0 / n_samples) * (X_train_opt.T @ error)
        grad_b = float((2.0 / n_samples) * np.sum(error))

        weights_opt -= lr * grad_w
        bias_opt -= lr * grad_b

    if needs_internal_scaling:
        weights = weights_opt / std
        bias = float(bias_opt - np.dot(mean, weights))
    else:
        weights = weights_opt
        bias = float(bias_opt)

    predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = _to_feature_matrix(data.train_features)
    y_train = _to_target_vector(data.train_target)
    X_test = _to_feature_matrix(data.test_features)

    n_samples, n_features = X_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)

    needs_internal_scaling = bool(np.max(np.abs(X_train)) > 15.0)
    if needs_internal_scaling:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        X_train_opt = (X_train - mean) / std
    else:
        X_train_opt = X_train

    weights_opt = np.zeros(n_features, dtype=float)
    bias_opt = 0.0
    loss_history: list[float] = []
    eps = 1e-15

    for _ in range(epochs):
        logits = X_train_opt @ weights_opt + bias_opt
        probas_train = _sigmoid(logits)

        clipped_probas = np.clip(probas_train, eps, 1.0 - eps)
        bce_loss = float(
            -np.mean(
                y_train * np.log(clipped_probas)
                + (1.0 - y_train) * np.log(1.0 - clipped_probas)
            )
        )
        loss_history.append(bce_loss)

        error = probas_train - y_train
        grad_w = (1.0 / n_samples) * (X_train_opt.T @ error)
        grad_b = float((1.0 / n_samples) * np.sum(error))

        weights_opt -= lr * grad_w
        bias_opt -= lr * grad_b

    if needs_internal_scaling:
        weights = weights_opt / std
        bias = float(bias_opt - np.dot(mean, weights))
    else:
        weights = weights_opt
        bias = float(bias_opt)

    probabilities = _sigmoid(X_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train = _to_feature_matrix(data.train_features)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = _to_feature_matrix(data.test_features)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    if np.max(np.abs(X_train)) > 15.0:
        scaler = StandardScaler()
        X_train_eval = scaler.fit_transform(X_train)
        X_test_eval = scaler.transform(X_test)
    else:
        X_train_eval = X_train
        X_test_eval = X_test

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_eval, y_train)
    lr_probas = lr_model.predict_proba(X_test_eval)[:, 1]
    lr_preds = lr_model.predict(X_test_eval)
    lr_metrics = _compute_classification_metrics(y_test, lr_probas, lr_preds)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_eval, y_train)
    dt_probas = dt_model.predict_proba(X_test_eval)[:, 1]
    dt_preds = dt_model.predict(X_test_eval)
    dt_metrics = _compute_classification_metrics(y_test, dt_probas, dt_preds)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_eval, y_train)
    knn_probas = knn_model.predict_proba(X_test_eval)[:, 1]
    knn_preds = knn_model.predict(X_test_eval)
    knn_metrics = _compute_classification_metrics(y_test, knn_probas, knn_preds)

    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )


def _metrics_to_dict(m: ClassificationMetrics) -> dict[str, float]:
    return {
        "roc_auc": round(float(m.roc_auc), 6),
        "accuracy": round(float(m.accuracy), 6),
        "precision": round(float(m.precision), 6),
        "recall": round(float(m.recall), 6),
        "f1": round(float(m.f1), 6),
    }


def run_experiments_and_save(base_dir: Path | None = None) -> dict[str, Any]:
    """Запускает обучение на Student_Performance.csv и german.csv и сохраняет experiment_results.json."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    df_student = pd.read_csv(base_dir / "Student_Performance.csv")
    df_student["Extracurricular Activities"] = df_student[
        "Extracurricular Activities"
    ].map({"Yes": 1, "No": 0})

    X_reg = df_student.drop("Performance Index", axis=1).to_numpy(dtype=float)
    y_reg = df_student["Performance Index"].to_numpy(dtype=float)

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    scaler_r = StandardScaler()
    X_tr_r_scaled = scaler_r.fit_transform(X_tr_r)
    X_te_r_scaled = scaler_r.transform(X_te_r)

    reg_input = RegressionInput(
        train_features=X_tr_r_scaled,
        train_target=y_tr_r,
        test_features=X_te_r_scaled,
        test_target=y_te_r,
        learning_rate=0.05,
        epochs=800,
    )
    reg_result = train_linear_regression(reg_input)
    test_mse = float(mean_squared_error(y_te_r, reg_result.predictions))

    df_german = pd.read_csv(base_dir / "german.csv", sep=";")
    target_col = df_german.columns[0]
    X_clf = df_german.drop(target_col, axis=1).to_numpy(dtype=float)
    y_clf = df_german[target_col].to_numpy(dtype=int)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    scaler_c = StandardScaler()
    X_tr_c_scaled = scaler_c.fit_transform(X_tr_c)
    X_te_c_scaled = scaler_c.transform(X_te_c)

    clf_input = ClassificationInput(
        train_features=X_tr_c_scaled,
        train_target=y_tr_c,
        test_features=X_te_c_scaled,
        test_target=y_te_c,
        learning_rate=0.1,
        epochs=1000,
    )
    own_clf_result = train_logistic_classifier(clf_input)
    own_metrics = _compute_classification_metrics(
        y_te_c, own_clf_result.probabilities, own_clf_result.predictions
    )
    baseline_results = compare_classical_models(clf_input)

    results_payload = {
        "linear_regression": {
            "test_mse": round(test_mse, 6),
        },
        "classification": {
            "own_logistic": _metrics_to_dict(own_metrics),
            "logistic_regression": _metrics_to_dict(
                baseline_results.logistic_regression
            ),
            "decision_tree": _metrics_to_dict(baseline_results.decision_tree),
            "knn": _metrics_to_dict(baseline_results.knn),
        },
    }

    output_path = base_dir / "experiment_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    return results_payload


if __name__ == "__main__":
    results = run_experiments_and_save()
    print(json.dumps(results, indent=2, ensure_ascii=False))