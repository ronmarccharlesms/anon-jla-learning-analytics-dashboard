"""Reproducible temporal-holdout evaluation for the FEUHS EDA models.

The script reads the same controlled CSV corpus as dashboard.py, rebuilds the
macro and micro feature sets, trains the deployed Random Forest specifications,
and prints a JSON report to stdout.  It never exports student-level rows.
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import analysis_engine as engine


CSV_FILES = [
    f"gb_{year}_{semester}.csv"
    for year in ('2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026')
    for semester in ('1', '2')
]


def _round(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _regression_metrics(actual, predicted):
    return {
        'n': int(len(actual)),
        'mae': _round(mean_absolute_error(actual, predicted)),
        'rmse': _round(math.sqrt(mean_squared_error(actual, predicted))),
        'r2': _round(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _classification_metrics(actual, probability, threshold):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        'threshold': threshold,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': _round(precision_score(actual, predicted, zero_division=0)),
        'recall': _round(recall_score(actual, predicted, zero_division=0)),
        'f1': _round(f1_score(actual, predicted, zero_division=0)),
    }


def _probability_metrics(actual, probability):
    if pd.Series(actual).nunique() < 2:
        return {'roc_auc': None, 'pr_auc': None, 'brier': None}
    return {
        'roc_auc': _round(roc_auc_score(actual, probability)),
        'pr_auc': _round(average_precision_score(actual, probability)),
        'brier': _round(brier_score_loss(actual, probability)),
    }


def _hard_rule_metrics(actual, grade_signal):
    predicted = (np.asarray(grade_signal) < engine.AT_RISK_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision': _round(precision_score(actual, predicted, zero_division=0)),
        'recall': _round(recall_score(actual, predicted, zero_division=0)),
        'f1': _round(f1_score(actual, predicted, zero_division=0)),
    }


def _calibration_bins(actual, probability, bins=5):
    frame = pd.DataFrame({'actual': np.asarray(actual), 'probability': probability})
    frame['bin'] = pd.qcut(frame['probability'], q=bins, duplicates='drop')
    grouped = frame.groupby('bin', observed=True).agg(
        n=('actual', 'size'),
        mean_probability=('probability', 'mean'),
        observed_rate=('actual', 'mean'),
    )
    return [
        {
            'n': int(row.n),
            'mean_probability': _round(row.mean_probability),
            'observed_rate': _round(row.observed_rate),
        }
        for row in grouped.itertuples()
    ]


def _subgroup_metrics(test, actual_col, prediction_col, group_cols):
    rows = []
    for keys, group in test.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metrics = _regression_metrics(group[actual_col], group[prediction_col])
        rows.append({**dict(zip(group_cols, map(str, keys))), **metrics})
    return rows


def evaluate_macro(data):
    features = engine.build_macro_features(data)
    available = [name for name in engine.MACRO_FEATURES if name in features]
    latest_year = int(features['year_int'].max())
    train = features[features['year_int'] < latest_year].dropna(subset=available).copy()
    test = features[features['year_int'] == latest_year].dropna(subset=available).copy()

    regressor = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    classifier = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    regressor.fit(train[available], train['mean_grade'])
    classifier.fit(train[available], train['at_risk'])
    test['rf_prediction'] = regressor.predict(test[available])
    probability = classifier.predict_proba(test[available])[:, 1]

    baselines = {}
    for column in ('prior_mean', 'strand_gwa'):
        baselines[column] = {
            'regression': _regression_metrics(test['mean_grade'], test[column]),
            'risk_rule': _hard_rule_metrics(test['at_risk'], test[column]),
        }

    return {
        'latest_year': latest_year,
        'train_n': int(len(train)),
        'test_n': int(len(test)),
        'risk_prevalence': _round(test['at_risk'].mean()),
        'random_forest': {
            'regression': _regression_metrics(test['mean_grade'], test['rf_prediction']),
            'probability': _probability_metrics(test['at_risk'], probability),
            'confusion_at_0_50': _classification_metrics(test['at_risk'], probability, 0.50),
            'confusion_at_0_60': _classification_metrics(test['at_risk'], probability, 0.60),
            'calibration': _calibration_bins(test['at_risk'], probability),
        },
        'baselines': baselines,
        'subgroups': _subgroup_metrics(
            test, 'mean_grade', 'rf_prediction', ['strand', 'grade_level', 'semester']
        ),
        'feature_importance': [
            {'feature': name, 'importance': _round(importance)}
            for name, importance in sorted(
                zip(available, regressor.feature_importances_),
                key=lambda item: item[1], reverse=True
            )
        ],
    }


def evaluate_micro(data, sample=None):
    features = engine.build_micro_features(data)
    available = [name for name in engine.MICRO_FEATURES if name in features]
    required = available + ['numeric_grade']
    if 'is_academic_grade_record' in features:
        features = features[features['is_academic_grade_record']]
    clean = features.dropna(subset=required).copy()
    if sample and len(clean) > sample:
        clean = clean.sample(sample, random_state=42)

    latest_year = int(clean['year_int'].max())
    train = clean[clean['year_int'] < latest_year].copy()
    test = clean[clean['year_int'] == latest_year].copy()

    regressor = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    classifier = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    regressor.fit(train[available], train['numeric_grade'])
    classifier.fit(train[available], train['at_risk'])
    test['rf_prediction'] = regressor.predict(test[available])
    probability = classifier.predict_proba(test[available])[:, 1]

    baselines = {}
    for column in ('prior_gwa', 'cumulative_gwa', 'peer_mean', 'subj_hist_mean'):
        baselines[column] = {
            'regression': _regression_metrics(test['numeric_grade'], test[column]),
            'risk_rule': _hard_rule_metrics(test['at_risk'], test[column]),
        }

    return {
        'latest_year': latest_year,
        'train_n': int(len(train)),
        'test_n': int(len(test)),
        'risk_prevalence': _round(test['at_risk'].mean()),
        'sampled': bool(sample),
        'random_forest': {
            'regression': _regression_metrics(test['numeric_grade'], test['rf_prediction']),
            'probability': _probability_metrics(test['at_risk'], probability),
            'confusion_at_0_50': _classification_metrics(test['at_risk'], probability, 0.50),
            'confusion_at_0_60': _classification_metrics(test['at_risk'], probability, 0.60),
            'calibration': _calibration_bins(test['at_risk'], probability),
        },
        'baselines': baselines,
        'subgroups': _subgroup_metrics(
            test, 'numeric_grade', 'rf_prediction', ['strand', 'grade_level', 'semester']
        ),
        'feature_importance': [
            {'feature': name, 'importance': _round(importance)}
            for name, importance in sorted(
                zip(available, regressor.feature_importances_),
                key=lambda item: item[1], reverse=True
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--micro-sample', type=int, default=None,
        help='Optional deterministic micro-row sample for quick smoke tests.'
    )
    args = parser.parse_args()
    started = time.time()
    data = engine.load_and_process_data(CSV_FILES)
    report = {
        'corpus': {
            'rows': int(len(data)),
            'students': int(data['student sis'].nunique()),
            'academic_grade_records': int(data['is_academic_grade_record'].sum()),
            'support_records': int(data['record_type'].eq('support').sum()),
            'school_years': sorted(map(str, data['school_year'].dropna().unique())),
        },
        'macro': evaluate_macro(data),
        'micro': evaluate_micro(data, sample=args.micro_sample),
    }
    report['elapsed_seconds'] = round(time.time() - started, 2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
