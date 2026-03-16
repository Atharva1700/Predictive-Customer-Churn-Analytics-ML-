"""
tests/test_model.py — Unit tests for model training and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data_generator import generate_churn_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import clean_data, encode_features, split_and_scale
from src.evaluate import compute_metrics
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def prepared_data(config):
    df = generate_churn_dataset(n=3000, random_state=42)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df, config)
    (X_train, X_test, X_res, y_res, y_train, y_test,
     feature_names, scaler) = split_and_scale(df, config)
    return X_train, X_test, y_train, y_test, feature_names


class TestModelTraining:

    def test_random_forest_trains(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_predictions_binary(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_probability_in_range(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[1] == 2
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_auc_above_random(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test, "Logistic Regression")
        assert metrics["auc_roc"] > 0.60, f"AUC {metrics['auc_roc']:.3f} not above random"

    def test_metrics_keys(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        for key in ["accuracy", "auc_roc", "f1", "avg_precision", "y_pred", "y_prob"]:
            assert key in metrics


class TestPredictPipeline:

    def test_predict_output_columns(self, config, prepared_data):
        from src.predict import predict, make_demo_sample
        X_train, X_test, y_train, y_test, feature_names = prepared_data

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        demo = make_demo_sample()
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Build minimal X for prediction
        X_scaled = X_test[:len(demo)]
        preds = predict(demo, model, X_scaled, config)

        assert "churn_probability" in preds.columns
        assert "churn_prediction" in preds.columns
        assert "risk_segment" in preds.columns

    def test_risk_segments_valid(self, config, prepared_data):
        from src.predict import predict, make_demo_sample
        X_train, X_test, y_train, y_test, _ = prepared_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        demo = make_demo_sample()
        X_scaled = X_test[:len(demo)]
        preds = predict(demo, model, X_scaled, config)

        valid_segments = {"Low Risk", "Medium Risk", "High Risk"}
        actual = set(preds["risk_segment"].dropna().astype(str))
        assert actual.issubset(valid_segments)
