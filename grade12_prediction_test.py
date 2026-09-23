"""Experimental external-grade to Grade 12 prediction study.

This module is intentionally separate from ``analysis_engine.py``.  It supports
an evaluation page only and must not be treated as production dashboard logic.

Method rationale follows established educational data-mining practice: compare
an interpretable baseline and regularized linear models with tree ensembles,
keep preprocessing inside validation folds, and report student-level held-out
performance.  Relevant methodological references are Romero & Ventura (2020),
"Educational data mining and learning analytics: An updated survey", and Baker
& Inventado (2014), "Educational Data Mining and Learning Analytics".  The
small cohort makes this an exploratory study; it does not support claims of
generalizable or causal prediction.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    r2_score,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel


EXTERNAL_SUBJECTS = ('Math', 'Science', 'English')
YEAR_TO_LEVEL = {'2024-2025': 'G11', '2025-2026': 'G12'}
SUPPORT_COURSES = {
    'Canvas Orientation Course For New Teachers',
    'FEU HS Student Clearance SY 2021-22',
    'Labster Simulations 2021-22',
}
FEATURE_COLUMNS = [
    'external_math', 'external_science', 'external_english', 'external_mean'
]
PROGRESSION_TOLERANCE = 2.0

# Provisional three-point reporting scale for the TEST study.  The anchor is
# explicit so the scale can be replaced after academic/registrar confirmation.
# Scores below 60 remain F; the project corpus has historically used 60 as its
# observed lower numeric bound.
LETTER_GRADE_BANDS = (
    (98, 100, 'A+'),
    (95, 97, 'A'),
    (92, 94, 'A-'),
    (89, 91, 'B+'),
    (86, 88, 'B'),
    (83, 85, 'B-'),
    (80, 82, 'C+'),
    (77, 79, 'C'),
    (74, 76, 'C-'),
    (71, 73, 'D'),
    (0, 70, 'F'),
)


def _external_id(value):
    number = pd.to_numeric(value, errors='coerce')
    if pd.isna(number) or float(number) % 1:
        return pd.NA
    return f'H{int(number)}'


def _normalise_course(value):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(value).lower())).strip()


def numeric_to_letter(value):
    """Convert a numeric grade using the provisional three-point scale."""
    number = pd.to_numeric(value, errors='coerce')
    if pd.isna(number):
        return pd.NA
    # Bands are defined by inclusive lower bounds. This handles decimal means
    # without gaps: 98.00 -> A+, 97.99 -> A, 94.99 -> A-, etc.
    for lower, _upper, letter in LETTER_GRADE_BANDS:
        if float(number) >= lower:
            return letter
    return pd.NA


def letter_grade_scale():
    """Return the visible conversion table used by the TEST study."""
    rows = []
    for index, (lower, upper, letter) in enumerate(LETTER_GRADE_BANDS):
        next_lower = LETTER_GRADE_BANDS[index - 1][0] if index else None
        display_upper = 100.00 if index == 0 else next_lower - 0.01
        rows.append({
            'Numeric range': f'{lower:.2f}-{display_upper:.2f}',
            'Letter grade': letter,
        })
    return pd.DataFrame(rows)


LETTER_TO_ORDINAL = {
    letter: ordinal
    for ordinal, letter in enumerate(
        ['F', 'D', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
    )
}


def numeric_to_letter_ordinal(value):
    letter = numeric_to_letter(value)
    return np.nan if pd.isna(letter) else LETTER_TO_ORDINAL[str(letter)]


def ordinal_to_letter(value):
    """Display the nearest ordered letter for an ordinal family mean."""
    number = pd.to_numeric(value, errors='coerce')
    if pd.isna(number):
        return pd.NA
    ordinal = int(np.clip(np.rint(float(number)), 0, max(LETTER_TO_ORDINAL.values())))
    return next(letter for letter, code in LETTER_TO_ORDINAL.items() if code == ordinal)


def build_letter_study_data(granular_data):
    """Create ordered-letter inputs and a three-class progression target."""
    data = granular_data.copy()
    grade_columns = [
        column for column in data.columns
        if column.startswith(('external_', 'feature_g11_subject_', 'feature_g11_family_', 'target_g12_subject_', 'target_g12_family_'))
        or column in {
            'feature_g11_overall_mean',
            'feature_g11_s1_mean', 'feature_g11_s2_mean',
            'target_g12_overall_mean'
        }
    ]
    for column in grade_columns:
        data[f'{column}_letter'] = data[column].map(numeric_to_letter)
        data[f'{column}_ordinal'] = data[column].map(numeric_to_letter_ordinal)

    external_ordinal_columns = [f'external_{subject}_ordinal' for subject in ('math', 'science', 'english')]
    data['external_baseline_ordinal'] = data[external_ordinal_columns].mean(axis=1)
    data['g12_overall_letter'] = data['target_g12_overall_mean'].map(numeric_to_letter)
    data['g12_overall_ordinal'] = data['target_g12_overall_mean'].map(numeric_to_letter_ordinal)
    movement = data['g12_overall_ordinal'] - data['external_baseline_ordinal']
    data['progression_status'] = np.select(
        [movement >= PROGRESSION_TOLERANCE, movement <= -PROGRESSION_TOLERANCE],
        ['Improved', 'Declined'], default='Maintained'
    )
    data.loc[data[['external_baseline_ordinal', 'g12_overall_ordinal']].isna().any(axis=1), 'progression_status'] = pd.NA

    # Preserve previously engineered context features. These are not grades:
    # variability, counts, strand, section, and missingness describe the record
    # structure and are valid classifier inputs alongside letter ordinals.
    ordinal_inputs = [
        column for column in data.columns
        if column.endswith('_ordinal') and (
            column.startswith('external_') or column.startswith('feature_g11_')
        )
    ]
    for column in ordinal_inputs:
        data[f'{column}_missing'] = data[column].isna().astype(float)
    return data


def letter_model_feature_columns(letter_data):
    """Return the ten pre-Grade-12 inputs used by the TEST classifiers."""
    feature_columns = [
        'external_math_ordinal', 'external_science_ordinal',
        'external_english_ordinal', 'external_mean_ordinal',
        'feature_g11_overall_mean_ordinal', 'feature_g11_s1_mean_ordinal',
        'feature_g11_s2_mean_ordinal', 'feature_g11_semester_change',
        'feature_g11_sd', 'feature_g11_course_count',
    ]
    return [column for column in feature_columns if column in letter_data]


def letter_feature_inventory(letter_data):
    """Describe every active letter-classifier input for the technical view."""
    rows = []
    for column in letter_model_feature_columns(letter_data):
        if column.startswith('external_') and column.endswith('_ordinal'):
            group = 'External JHS inputs'
            meaning = 'External JHS Math, Science, English, or overall ordinal'
            construction = 'Numeric external grade converted to ordered letter code'
        elif column == 'feature_g11_overall_mean_ordinal':
            group = 'Grade 11 overall summary'
            meaning = 'Overall retained Grade 11 achievement'
            construction = 'Grade 11 overall mean converted to ordered letter code'
        elif column in {'feature_g11_s1_mean_ordinal', 'feature_g11_s2_mean_ordinal'}:
            group = 'Grade 11 semester performance'
            meaning = 'Grade 11 semester-level achievement'
            construction = 'Semester numeric mean converted to ordered letter code'
        elif column == 'feature_g11_semester_change':
            group = 'Grade 11 progression'
            meaning = 'Change from Grade 11 Semester 1 to Semester 2'
            construction = 'Semester 2 numeric mean minus Semester 1 numeric mean'
        elif column == 'feature_g11_sd':
            group = 'Grade 11 variability'
            meaning = 'Consistency across retained Grade 11 courses'
            construction = 'Standard deviation of retained Grade 11 course grades'
        elif column == 'feature_g11_course_count':
            group = 'Grade 11 course coverage'
            meaning = 'Number of retained Grade 11 courses available'
            construction = 'Distinct retained Grade 11 course count'
        else:
            group = 'Other active input'
            meaning = 'Active classifier predictor'
            construction = 'Derived from the retained Grade 11 record'
        rows.append({
            'Feature': column,
            'Feature group': group,
            'Educational meaning': meaning,
            'Construction': construction,
        })
    return pd.DataFrame(rows)


def evaluate_letter_models(letter_data):
    """Evaluate probability-averaged letter progression classifiers."""
    feature_columns = letter_model_feature_columns(letter_data)
    clean = letter_data[['student sis', 'external_baseline_ordinal', 'g12_overall_letter', 'progression_status', *feature_columns]].dropna(
        subset=['student sis', 'external_baseline_ordinal', 'g12_overall_letter', 'progression_status']
    ).reset_index(drop=True)
    if len(clean) < 10 or clean['progression_status'].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), 'Insufficient letter progression classes.'

    classes = np.array(['Declined', 'Maintained', 'Improved'])
    factories = {
        'Regularized Multinomial Logistic': lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, multi_class='multinomial', random_state=42)
        ),
        'Random Forest Classifier': lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=1
        ),
        'Extra Trees Classifier': lambda: ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=1
        ),
    }
    model_names = ['Ordinal Logistic Regression', *factories]
    probabilities = {name: np.full((len(clean), len(classes)), np.nan) for name in model_names}
    for test_index in range(len(clean)):
        train_indices = np.delete(np.arange(len(clean)), test_index)
        train_X = clean[feature_columns].iloc[train_indices].replace([np.inf, -np.inf], np.nan)
        test_X = clean[feature_columns].iloc[[test_index]].replace([np.inf, -np.inf], np.nan)
        medians = train_X.median().fillna(0.0)
        y_train = clean['progression_status'].iloc[train_indices]
        ordinal_train = train_X.fillna(medians).to_numpy(dtype=float)
        ordinal_test = test_X.fillna(medians).to_numpy(dtype=float)
        ordinal_scaler = StandardScaler().fit(train_X.fillna(medians).to_numpy(dtype=float))
        ordinal_train = ordinal_scaler.transform(train_X.fillna(medians).to_numpy(dtype=float))
        ordinal_test = ordinal_scaler.transform(test_X.fillna(medians).to_numpy(dtype=float))
        # OrderedModel is unregularized; retain nonconstant predictors and use
        # a compact latent representation to avoid an underidentified fit in
        # the current small cohort. PCA is fitted within this fold.
        variance = np.nanstd(ordinal_train, axis=0)
        keep = variance > 1e-12
        ordinal_train = ordinal_train[:, keep]
        ordinal_test = ordinal_test[:, keep]
        if ordinal_train.shape[1] > 8:
            pca = PCA(n_components=min(8, ordinal_train.shape[0] - 3), random_state=42)
            ordinal_train = pca.fit_transform(ordinal_train)
            ordinal_test = pca.transform(ordinal_test)
        y_codes = y_train.map({'Declined': 0, 'Maintained': 1, 'Improved': 2}).to_numpy()
        if np.unique(y_codes).size >= 2:
            try:
                ordinal_model = OrderedModel(y_codes, ordinal_train, distr='logit')
                ordinal_fit = ordinal_model.fit(method='bfgs', disp=False, maxiter=300)
                ordinal_probability = np.asarray(ordinal_fit.model.predict(ordinal_fit.params, exog=ordinal_test)[0])
                ordinal_classes = np.unique(y_codes)
                for class_code, probability in zip(ordinal_classes, ordinal_probability):
                    probabilities['Ordinal Logistic Regression'][test_index, class_code] = probability
            except Exception:
                pass
        for name, factory in factories.items():
            model = factory()
            model.fit(train_X.fillna(medians), y_train)
            model_classes = model.classes_
            fold_probability = model.predict_proba(test_X.fillna(medians))[0]
            for class_name, probability in zip(model_classes, fold_probability):
                probabilities[name][test_index, np.where(classes == class_name)[0][0]] = probability

    probability_frame = clean[['student sis', 'external_baseline_ordinal', 'g12_overall_letter']].copy()
    for name, values in probabilities.items():
        for index, class_name in enumerate(classes):
            probability_frame[f'{name} - P({class_name})'] = values[:, index]
    model_probability_columns = {
        name: [f'{name} - P({class_name})' for class_name in classes]
        for name in model_names
    }
    probability_frame['Consensus P(Declined)'] = probability_frame[[columns[0] for columns in model_probability_columns.values()]].mean(axis=1)
    probability_frame['Consensus P(Maintained)'] = probability_frame[[columns[1] for columns in model_probability_columns.values()]].mean(axis=1)
    probability_frame['Consensus P(Improved)'] = probability_frame[[columns[2] for columns in model_probability_columns.values()]].mean(axis=1)
    consensus_columns = ['Consensus P(Declined)', 'Consensus P(Maintained)', 'Consensus P(Improved)']
    probability_frame['Predicted Status'] = np.array(['Declined', 'Maintained', 'Improved'])[
        probability_frame[consensus_columns].to_numpy().argmax(axis=1)
    ]
    probability_frame['Probability Consensus - Predicted Status'] = probability_frame['Predicted Status']
    for name in model_names:
        model_probability_columns = [f'{name} - P({class_name})' for class_name in classes]
        probability_frame[f'{name} - Predicted Status'] = np.array(
            ['Declined', 'Maintained', 'Improved']
        )[probability_frame[model_probability_columns].to_numpy().argmax(axis=1)]
    probability_frame['Confidence'] = probability_frame[consensus_columns].max(axis=1)
    probability_frame['Assessment'] = np.where(
        probability_frame['Confidence'] >= 0.60,
        'Likely ' + probability_frame['Predicted Status'],
        'Uncertain'
    )
    rows = []
    for name in model_names:
        predicted = np.array(['Declined', 'Maintained', 'Improved'])[
            probabilities[name].argmax(axis=1)
        ]
        rows.append({
            'Model': name,
            'N': len(clean),
            'Accuracy': accuracy_score(clean['progression_status'], predicted),
            'Balanced Accuracy': balanced_accuracy_score(clean['progression_status'], predicted),
            'Macro F1': f1_score(clean['progression_status'], predicted, average='macro', zero_division=0),
        })
    consensus_predicted = probability_frame['Predicted Status']
    rows.append({
        'Model': 'Probability Consensus',
        'N': len(clean),
        'Accuracy': accuracy_score(clean['progression_status'], consensus_predicted),
        'Balanced Accuracy': balanced_accuracy_score(clean['progression_status'], consensus_predicted),
        'Macro F1': f1_score(clean['progression_status'], consensus_predicted, average='macro', zero_division=0),
    })
    metrics = pd.DataFrame(rows).sort_values('Macro F1', ascending=False).reset_index(drop=True)
    probability_frame['Actual Status'] = clean['progression_status'].to_numpy()
    return metrics, probability_frame, None


def letter_classification_diagnostics(predictions):
    """Return per-class recall and confusion matrices for held-out classifications."""
    labels = ['Declined', 'Maintained', 'Improved']
    model_names = [
        'Ordinal Logistic Regression',
        'Regularized Multinomial Logistic',
        'Random Forest Classifier',
        'Extra Trees Classifier',
        'Probability Consensus',
    ]
    recall_rows = []
    confusion_rows = []
    actual = predictions['Actual Status']
    for model_name in model_names:
        column = 'Predicted Status' if model_name == 'Probability Consensus' else f'{model_name} - Predicted Status'
        predicted = predictions[column]
        recalls = recall_score(actual, predicted, labels=labels, average=None, zero_division=0)
        recall_rows.extend({
            'Model': model_name,
            'Class': label,
            'Recall': score,
        } for label, score in zip(labels, recalls))
        matrix = confusion_matrix(actual, predicted, labels=labels)
        for actual_label, row in zip(labels, matrix):
            for predicted_label, count in zip(labels, row):
                confusion_rows.append({
                    'Model': model_name,
                    'Actual': actual_label,
                    'Predicted': predicted_label,
                    'Students': int(count),
                })
    return pd.DataFrame(recall_rows), pd.DataFrame(confusion_rows)


def _root_csv_paths(root):
    return sorted(Path(root).glob('gb_*.csv'))


def _external_workbook_paths(root):
    candidates = [
        Path(root) / 'tests' / 'first_100.xlsx',
        Path(root) / 'tests' / 'jhs_g9_sampling.xlsx',
    ]
    return [path for path in candidates if path.exists()]


def _read_external_workbook(path):
    workbook = pd.ExcelFile(path)
    if 'Sheet1' in workbook.sheet_names:
        sheet_name = 'Sheet1'
    elif 'Sheet2' in workbook.sheet_names:
        sheet_name = 'Sheet2'
    else:
        sheet_name = workbook.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet_name)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    required = {'student number', 'math', 'science', 'english'}
    if not required.issubset(columns):
        missing = ', '.join(sorted(required - set(columns)))
        raise ValueError(f'External workbook {path.name} is missing columns: {missing}')
    return pd.DataFrame({
        'student sis': frame[columns['student number']].map(_external_id),
        'external_math': pd.to_numeric(frame[columns['math']], errors='coerce'),
        'external_science': pd.to_numeric(frame[columns['science']], errors='coerce'),
        'external_english': pd.to_numeric(frame[columns['english']], errors='coerce'),
    }).dropna(subset=['student sis'])


def load_study_data(root='.'):
    """Link the external workbook to approximate G11/G12 overall outcomes."""
    root = Path(root)
    external_paths = _external_workbook_paths(root)
    if not external_paths:
        raise FileNotFoundError('No external JHS workbook was found in tests/.')
    external_frames = []
    for path in external_paths:
        source = _read_external_workbook(path)
        source = source.groupby('student sis', as_index=False).mean(numeric_only=True)
        external_frames.append(source)
    # The expanded first_100 workbook is first, so overlapping students are
    # represented once without giving either source extra weight.
    external = pd.concat(external_frames, ignore_index=True).drop_duplicates(
        subset=['student sis'], keep='first'
    )

    # A row of zeroes is treated as an invalid/missing external record for this
    # exploratory study; zero is outside the observed FEUHS grade convention.
    zero_row = external[list(FEATURE_COLUMNS[:3])].eq(0).all(axis=1)
    external.loc[zero_row, list(FEATURE_COLUMNS[:3])] = np.nan
    external['external_mean'] = external[list(FEATURE_COLUMNS[:3])].mean(axis=1)
    external['external_family_mathematics'] = external['external_math']
    external['external_family_science'] = external['external_science']
    external['external_family_english_language'] = external['external_english']
    external['external_family_mean'] = external[
        ['external_family_mathematics', 'external_family_science',
         'external_family_english_language']
    ].mean(axis=1)

    frames = []
    for path in _root_csv_paths(root):
        frame = pd.read_csv(path, low_memory=False)
        frame['study_year'] = path.stem.removeprefix('gb_').rsplit('_', 1)[0]
        frames.append(frame[['student sis', 'course', 'unposted final grade', 'study_year']])
    if not frames:
        raise FileNotFoundError('No gb_*.csv files were found in the repository root.')

    grades = pd.concat(frames, ignore_index=True)
    grades['student sis'] = grades['student sis'].astype('string').str.strip()
    grades['numeric_grade'] = pd.to_numeric(
        grades['unposted final grade'], errors='coerce'
    )
    grades['course_key'] = grades['course'].map(_normalise_course)
    grades = grades[
        grades['student sis'].isin(set(external['student sis'].dropna()))
        & grades['study_year'].isin(YEAR_TO_LEVEL)
        & grades['numeric_grade'].between(0, 100)
        & ~grades['course'].isin(SUPPORT_COURSES)
    ].copy()

    # Average repeated records within student/course/year before computing an
    # overall mean, preventing section irregularity from overweighting a course.
    course_means = (
        grades.groupby(['student sis', 'study_year', 'course_key'], as_index=False)
        ['numeric_grade'].mean()
    )
    overall = (
        course_means.assign(level=course_means['study_year'].map(YEAR_TO_LEVEL))
        .groupby(['student sis', 'level'], as_index=False)
        .agg(overall_mean=('numeric_grade', 'mean'), course_count=('course_key', 'nunique'))
        .pivot(index='student sis', columns='level', values=['overall_mean', 'course_count'])
    )
    overall.columns = [f'{metric.lower()}_{level.lower()}' for metric, level in overall.columns]
    overall = overall.reset_index()

    result = external.merge(overall, on='student sis', how='left')
    return result


def descriptive_summary(data):
    """Return compact descriptive statistics for the study page."""
    rows = []
    columns = FEATURE_COLUMNS + ['overall_mean_g11', 'overall_mean_g12']
    labels = {
        'external_math': 'External Math',
        'external_science': 'External Science',
        'external_english': 'External English',
        'external_mean': 'External Mean',
        'overall_mean_g11': 'Grade 11 Overall',
        'overall_mean_g12': 'Grade 12 Overall',
    }
    for column in columns:
        values = pd.to_numeric(data[column], errors='coerce').dropna()
        rows.append({
            'Measure': labels[column],
            'N': int(values.size),
            'Missing': int(data[column].isna().sum()),
            'Mean': values.mean(),
            'SD': values.std(ddof=1),
            'Minimum': values.min(),
            'Median': values.median(),
            'Maximum': values.max(),
        })
    return pd.DataFrame(rows)


def correlation_summary(data):
    """Return Pearson and Spearman associations for descriptive inspection."""
    rows = []
    for feature in FEATURE_COLUMNS:
        for target in ('overall_mean_g11', 'overall_mean_g12'):
            pair = data[[feature, target]].dropna()
            if len(pair) < 3:
                pearson = spearman = np.nan
            else:
                pearson = pearsonr(pair[feature], pair[target]).statistic
                spearman = spearmanr(pair[feature], pair[target]).statistic
            rows.append({
                'Predictor': feature.replace('external_', 'External ').title(),
                'Target': target.replace('overall_mean_', '').upper() + ' Overall',
                'N': len(pair),
                'Pearson r': pearson,
                'Spearman rho': spearman,
            })
    return pd.DataFrame(rows)


def _models():
    """Return fixed, literature-grounded tabular model candidates."""
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elastic = make_pipeline(
        StandardScaler(), ElasticNet(alpha=0.15, l1_ratio=0.5, max_iter=10000)
    )
    forest = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, max_features=1.0,
        random_state=42, n_jobs=1
    )
    extra = ExtraTreesRegressor(
        n_estimators=200, min_samples_leaf=3, max_features=1.0,
        random_state=42, n_jobs=1
    )
    boosting = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=1.0, random_state=42
    )
    ensemble = VotingRegressor([
        ('ridge', ridge), ('random_forest', forest), ('extra_trees', extra)
    ])
    return {
        'Ridge': ridge,
        'Elastic Net': elastic,
        'Random Forest': forest,
        'Extra Trees': extra,
        'Histogram Gradient Boosting': boosting,
        'Voting Ensemble': ensemble,
    }


def evaluate_models(data):
    """Evaluate fixed models with leave-one-student-out predictions."""
    clean = data[
        ['student sis', *FEATURE_COLUMNS, 'overall_mean_g11', 'overall_mean_g12']
    ].dropna(subset=['student sis', 'external_mean', 'overall_mean_g12']).reset_index(drop=True)
    if len(clean) < 10:
        return pd.DataFrame(), pd.DataFrame(), 'Fewer than 10 complete G12 cases.'

    X = clean[FEATURE_COLUMNS]
    y = clean['overall_mean_g12']
    predictions = {'Actual G12': y.to_numpy()}
    model_names = ['Mean Baseline', *_models().keys()]
    for name in model_names:
        predictions[name] = np.full(len(clean), np.nan)

    for test_index in range(len(clean)):
        train_indices = np.delete(np.arange(len(clean)), test_index)
        X_train, X_test = X.iloc[train_indices], X.iloc[[test_index]]
        y_train = y.iloc[train_indices]
        predictions['Mean Baseline'][test_index] = y_train.mean()
        for name, model in _models().items():
            model.fit(X_train, y_train)
            predictions[name][test_index] = model.predict(X_test)[0]

    prediction_frame = clean[
        ['student sis', *FEATURE_COLUMNS, 'overall_mean_g11']
    ].copy()
    prediction_frame['Actual G12'] = y.to_numpy()
    for name in model_names:
        prediction_frame[name] = predictions[name]
    rows = []
    for name in model_names:
        actual = prediction_frame['Actual G12']
        predicted = prediction_frame[name]
        rows.append({
            'Model': name,
            'N': len(actual),
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'R2': r2_score(actual, predicted),
            'Spearman rho': spearmanr(actual, predicted).statistic,
        })
    metrics = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    return metrics, prediction_frame, None


# Granular study layer -------------------------------------------------------
# The selected cohort remains one row per student.  Course-level Grade 11
# values become features, while Grade 12 course values remain separate targets.
FAMILY_RULES = {
    'science': ('science', 'physics', 'chemistry', 'biology', 'earth science', 'environmental'),
    'mathematics': (
        'math', 'mathematics', 'statistics', 'calculus',
        'empowerment technologies',
    ),
    'english_language': ('english', 'reading', 'writing', 'literature'),
}
RETAINED_FAMILIES = frozenset(FAMILY_RULES)


def _subject_family(course_key, strand=None):
    """Map only retained families; return None for excluded subject domains.

    STEM and non-STEM General Mathematics/Statistics variants are both retained
    as Mathematics.  ``strand`` is accepted so the feature layer can preserve
    the student's strand-specific course choice without mixing course labels.
    """
    if 'filipino sa piling larang' in course_key or 'komunikasyon' in course_key:
        return None
    for family, terms in FAMILY_RULES.items():
        if any(term in course_key for term in terms):
            return family
    return None


def _parse_strand(section):
    value = '' if pd.isna(section) else str(section).upper()
    match = re.search(r'(?:^|_)(11|12)([SAHG])', value)
    if not match:
        return pd.NA
    return {'S': 'STEM', 'A': 'ABM', 'H': 'HUMSS', 'G': 'GAS'}.get(match.group(2), pd.NA)


def build_granular_study_data(root='.'):
    """Build one student row with granular G11 features and G12 targets.

    Section-derived fields are deliberately limited to count/diversity proxies.
    The available section values do not establish a registrar-confirmed usual
    section, so these are not labelled as definitive irregular-student flags.
    """
    base = load_study_data(root)
    root = Path(root)
    frames = []
    for path in _root_csv_paths(root):
        frame = pd.read_csv(path, low_memory=False)
        frame['study_year'] = path.stem.removeprefix('gb_').rsplit('_', 1)[0]
        frame['study_semester'] = path.stem.rsplit('_', 1)[1]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw['student sis'] = raw['student sis'].astype('string').str.strip()
    raw['numeric_grade'] = pd.to_numeric(raw['unposted final grade'], errors='coerce')
    raw['course_key'] = raw['course'].map(_normalise_course)
    raw['student_strand'] = raw['section sis'].map(_parse_strand)
    raw = raw[
        raw['student sis'].isin(set(base['student sis'].dropna()))
        & raw['study_year'].isin(YEAR_TO_LEVEL)
        & raw['numeric_grade'].between(0, 100)
        & ~raw['course'].isin(SUPPORT_COURSES)
    ].copy()

    # Keep the strand-specific Math/Statistics course that matches the student
    # where the section encoding is parseable. Unknown strand values retain the
    # row for auditability rather than silently discarding it.
    stem_variant = raw['course_key'].str.contains(r'\bstem\b', na=False)
    non_stem_variant = raw['course_key'].str.contains(r'\bnon stem\b', na=False)
    raw = raw[
        (~stem_variant | raw['student_strand'].isna() | raw['student_strand'].eq('STEM'))
        & (~non_stem_variant | raw['student_strand'].isna() | ~raw['student_strand'].eq('STEM'))
    ].copy()

    course_means = (
        raw.groupby(['student sis', 'study_year', 'study_semester', 'course_key'], as_index=False)
        .agg(numeric_grade=('numeric_grade', 'mean'))
    )
    g11 = course_means[course_means['study_year'].eq('2024-2025')].copy()
    g12 = course_means[course_means['study_year'].eq('2025-2026')].copy()
    g11_retained = g11[g11['course_key'].map(_subject_family).notna()].copy()
    g12_retained = g12[g12['course_key'].map(_subject_family).notna()].copy()
    external_family_columns = [
        'external_family_mathematics', 'external_family_science',
        'external_family_english_language', 'external_family_mean'
    ]
    result = base[
        ['student sis', *FEATURE_COLUMNS, 'overall_mean_g11', *external_family_columns]
    ].copy()
    result['feature_g11_overall_mean'] = result['overall_mean_g11']

    def add_pivot(source, level, prefix):
        pivot = source.pivot_table(
            index='student sis', columns='course_key', values='numeric_grade', aggfunc='mean'
        ).add_prefix(prefix)
        return pivot.reset_index()

    result = result.merge(
        add_pivot(g11_retained, 'G11', 'feature_g11_subject_'),
        on='student sis', how='left'
    )
    target_pivot = add_pivot(g12_retained, 'G12', 'target_g12_subject_')
    result = result.merge(target_pivot, on='student sis', how='left')

    family_source = g11_retained.assign(
        family=g11_retained['course_key'].map(_subject_family)
    )
    family_values = family_source.groupby(['student sis', 'family'])['numeric_grade'].mean().unstack()
    family_values = family_values.add_prefix('feature_g11_family_').reset_index()
    result = result.merge(family_values, on='student sis', how='left')

    g12_family_values = (
        g12_retained.assign(family=g12_retained['course_key'].map(_subject_family))
        .groupby(['student sis', 'family'])['numeric_grade']
        .mean()
        .unstack()
        .add_prefix('target_g12_family_')
        .reset_index()
    )
    result = result.merge(g12_family_values, on='student sis', how='left')

    semester_values = g11_retained.groupby(['student sis', 'study_semester'])['numeric_grade'].mean().unstack()
    semester_values = semester_values.rename(columns={
        '1': 'feature_g11_s1_mean', '2': 'feature_g11_s2_mean'
    }).reset_index()
    result = result.merge(semester_values, on='student sis', how='left')
    result['feature_g11_semester_change'] = (
        result.get('feature_g11_s2_mean', np.nan) - result.get('feature_g11_s1_mean', np.nan)
    )

    g11_stats = g11_retained.groupby('student sis')['numeric_grade'].agg(
        feature_g11_sd='std', feature_g11_min='min', feature_g11_max='max',
        feature_g11_range=lambda values: values.max() - values.min(),
        feature_g11_course_count='nunique'
    ).reset_index()
    result = result.merge(g11_stats, on='student sis', how='left')

    section_rows = raw[raw['study_year'].eq('2024-2025')].copy()
    section_rows['strand_value'] = section_rows['section sis'].map(_parse_strand)
    section_stats = section_rows.groupby('student sis').agg(
        feature_g11_section_count=('section sis', 'nunique'),
        feature_g11_section_rows=('section sis', 'size'),
        feature_g11_strand=('strand_value', lambda values: values.dropna().mode().iloc[0] if not values.dropna().empty else 'Unknown')
    ).reset_index()
    result = result.merge(section_stats, on='student sis', how='left')
    result['feature_g11_section_count'] = result['feature_g11_section_count'].fillna(0)
    result['feature_g11_strand'] = result['feature_g11_strand'].fillna('Unknown')
    result['feature_g11_strand_STEM'] = result['feature_g11_strand'].eq('STEM').astype(float)
    result['feature_g11_strand_ABM'] = result['feature_g11_strand'].eq('ABM').astype(float)
    result['feature_g11_strand_HUMSS'] = result['feature_g11_strand'].eq('HUMSS').astype(float)
    result['feature_g11_strand_GAS'] = result['feature_g11_strand'].eq('GAS').astype(float)
    result = result.drop(columns=['feature_g11_strand'])

    target_means = g12.groupby('student sis')['numeric_grade'].mean().rename('target_g12_overall_mean')
    result = result.merge(target_means, on='student sis', how='left')
    result['target_g12_gain'] = (
        result['target_g12_overall_mean'] - result['external_mean']
    )
    return result


def _granular_feature_columns(data):
    return [
        column for column in data.columns
        if column.startswith('feature_') or column in FEATURE_COLUMNS
    ]


def evaluate_granular_models(data):
    """Evaluate granular features against the Grade 12 overall target.

    Ridge and PLS are primary low-sample models; tree ensembles are sensitivity
    analyses. Every fold imputes and scales only its training observations.
    """
    feature_columns = _granular_feature_columns(data)
    numeric_features = [
        column for column in data[feature_columns].select_dtypes(include='number').columns
        if data[column].notna().any()
    ]
    clean = data[['student sis', *numeric_features, 'target_g12_overall_mean']].dropna(
        subset=['student sis', 'external_mean', 'target_g12_overall_mean']
    ).reset_index(drop=True)
    if len(clean) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 'Fewer than 10 complete G12 cases.'
    X = clean[numeric_features].replace([np.inf, -np.inf], np.nan)
    y = clean['target_g12_overall_mean']
    model_factories = {
        'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Elastic Net': lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.15, l1_ratio=0.5, max_iter=10000)),
        'PLS': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=min(2, X.shape[1]))),
        'Extra Trees': lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=1),
    }
    predictions = {
        'student sis': clean['student sis'].to_numpy(),
        'external_mean': clean['external_mean'].to_numpy(),
        'Actual G12': y.to_numpy(),
    }
    for name in model_factories:
        predictions[name] = np.full(len(clean), np.nan)
    predictions['Mean Baseline'] = np.full(len(clean), np.nan)
    for test_index in range(len(clean)):
        train_indices = np.delete(np.arange(len(clean)), test_index)
        predictions['Mean Baseline'][test_index] = y.iloc[train_indices].mean()
        train_X = X.iloc[train_indices].copy()
        test_X = X.iloc[[test_index]].copy()
        train_medians = train_X.median().fillna(y.iloc[train_indices].mean())
        train_X = train_X.fillna(train_medians)
        test_X = test_X.fillna(train_medians)
        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_X, y.iloc[train_indices])
            predictions[name][test_index] = float(np.asarray(model.predict(test_X)).ravel()[0])
    prediction_frame = pd.DataFrame(predictions)
    metric_rows = []
    tolerance = PROGRESSION_TOLERANCE
    actual_gain = y.to_numpy() - clean['external_mean'].to_numpy()
    actual_status = np.where(
        actual_gain >= tolerance, 'Improved',
        np.where(actual_gain <= -tolerance, 'Declined', 'Maintained')
    )
    for name in ['Mean Baseline', *model_factories]:
        predicted_gain = prediction_frame[name].to_numpy() - clean['external_mean'].to_numpy()
        predicted_status = np.where(
            predicted_gain >= tolerance, 'Improved',
            np.where(predicted_gain <= -tolerance, 'Declined', 'Maintained')
        )
        metric_rows.append({
            'Model': name, 'N': len(y),
            'MAE': mean_absolute_error(y, prediction_frame[name]),
            'RMSE': np.sqrt(mean_squared_error(y, prediction_frame[name])),
            'R2': r2_score(y, prediction_frame[name]),
            'Spearman rho': spearmanr(y, prediction_frame[name]).statistic,
            'Gain MAE': mean_absolute_error(actual_gain, predicted_gain),
            'Status Macro F1': f1_score(actual_status, predicted_status, average='macro', zero_division=0),
            'Status Balanced Accuracy': balanced_accuracy_score(actual_status, predicted_status),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(
        ['Status Macro F1', 'Status Balanced Accuracy', 'Gain MAE', 'MAE', 'RMSE'],
        ascending=[False, False, True, True, True]
    ).reset_index(drop=True)
    metrics['Selected Primary'] = False
    metrics.loc[0, 'Selected Primary'] = True

    target_rows = []
    target_columns = [
        column for column in data
        if column.startswith('target_g12_subject_') or column.startswith('target_g12_family_')
    ]
    for target in target_columns:
        target_data = data[['student sis', *numeric_features, target]].dropna(subset=[target])
        if len(target_data) < 10:
            continue
        target_y = target_data[target]
        target_X = target_data[numeric_features].replace([np.inf, -np.inf], np.nan)
        oof = np.full(len(target_data), np.nan)
        for test_index in range(len(target_data)):
            train_indices = np.delete(np.arange(len(target_data)), test_index)
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            train_X = target_X.iloc[train_indices].copy()
            test_X = target_X.iloc[[test_index]].copy()
            train_medians = train_X.median().fillna(target_y.iloc[train_indices].mean())
            model.fit(train_X.fillna(train_medians), target_y.iloc[train_indices])
            oof[test_index] = float(np.asarray(model.predict(test_X.fillna(train_medians))).ravel()[0])
        target_rows.append({
            'Target': target.replace('target_g12_subject_', '').replace('target_g12_family_', ''),
            'Target Type': 'Subject' if target.startswith('target_g12_subject_') else 'Family',
            'N': len(target_data),
            'MAE': mean_absolute_error(target_y, oof),
            'RMSE': np.sqrt(mean_squared_error(target_y, oof)),
            'R2': r2_score(target_y, oof),
        })
    subject_metrics = pd.DataFrame(target_rows).sort_values('MAE') if target_rows else pd.DataFrame()
    return metrics, prediction_frame, subject_metrics, None


def granular_feature_importance(data, model_name):
    """Fit one selected model to the complete cohort for explanatory ranking.

    This is not validation output. It explains which retained predictors the
    selected full-cohort model used after the out-of-fold model comparison.
    """
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.loc[:, ~data.columns.duplicated()].copy()
    feature_columns = list(dict.fromkeys(_granular_feature_columns(data)))
    numeric_features = list(dict.fromkeys(
        column for column in data[feature_columns].select_dtypes(include='number').columns
        if data[column].notna().any()
    ))
    clean = data[['student sis', *numeric_features, 'external_mean', 'target_g12_overall_mean']].dropna(
        subset=['student sis', 'external_mean', 'target_g12_overall_mean']
    )
    clean = clean.loc[:, ~clean.columns.duplicated()].copy()
    if clean.empty:
        return pd.DataFrame()
    X = clean[numeric_features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median().fillna(clean['target_g12_overall_mean'].mean()))
    y = clean['target_g12_overall_mean']
    factories = {
        'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Elastic Net': lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.15, l1_ratio=0.5, max_iter=10000)),
        'PLS': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=min(2, X.shape[1]))),
        'Extra Trees': lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=1),
    }
    if model_name not in factories:
        return pd.DataFrame()
    model = factories[model_name]()
    model.fit(X, y)
    estimator = model[-1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'feature_importances_'):
        importance = np.asarray(estimator.feature_importances_)
        label = 'Tree importance'
    elif hasattr(estimator, 'coef_'):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()
        label = 'Absolute coefficient'
    else:
        return pd.DataFrame()
    return pd.DataFrame({
        'Feature': numeric_features,
        'Importance': importance,
        'Importance Type': label,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)


def summarize_model_consensus(predictions, model_names, tolerance=PROGRESSION_TOLERANCE):
    """Summarize per-student direct predictions for administrator interpretation."""
    result = predictions[['student sis', 'external_mean', *model_names]].copy()
    result['Consensus G12'] = predictions[list(model_names)].mean(axis=1)
    result['Prediction Spread'] = predictions[list(model_names)].max(axis=1) - predictions[list(model_names)].min(axis=1)
    result['Predicted Gain'] = result['Consensus G12'] - result['external_mean']
    result['Assessment'] = np.select(
        [
            (result['Predicted Gain'] >= tolerance) & (result['Prediction Spread'] <= tolerance),
            (result['Predicted Gain'] <= -tolerance) & (result['Prediction Spread'] <= tolerance),
            (result['Predicted Gain'].abs() < tolerance) & (result['Prediction Spread'] <= tolerance),
        ],
        ['Likely Improved', 'Likely Declined', 'Likely Maintained'],
        default='Uncertain',
    )
    return result


def build_student_summary(granular_data, consensus, student_sis):
    """Return a compact administrator-facing summary for one student."""
    row = granular_data[granular_data['student sis'].eq(student_sis)]
    model_row = consensus[consensus['student sis'].eq(student_sis)]
    if row.empty or model_row.empty:
        return {}
    row = row.iloc[0]
    model_row = model_row.iloc[0]

    subject_values = {
        column.removeprefix('target_g12_subject_'): row[column]
        for column in granular_data.columns
        if column.startswith('target_g12_subject_') and pd.notna(row[column])
    }
    family_values = {
        column.removeprefix('target_g12_family_'): row[column]
        for column in granular_data.columns
        if column.startswith('target_g12_family_') and pd.notna(row[column])
    }
    strongest_subject = max(subject_values.items(), key=lambda item: item[1], default=(None, np.nan))
    weakest_subject = min(subject_values.items(), key=lambda item: item[1], default=(None, np.nan))
    strongest_family = max(family_values.items(), key=lambda item: item[1], default=(None, np.nan))
    return {
        'External baseline': model_row['external_mean'],
        'Consensus predicted G12': model_row['Consensus G12'],
        'Predicted gain': model_row['Predicted Gain'],
        'Assessment': model_row['Assessment'],
        'Model spread': model_row['Prediction Spread'],
        'Strongest subject': strongest_subject[0],
        'Strongest subject grade': strongest_subject[1],
        'Weakest subject': weakest_subject[0],
        'Weakest subject grade': weakest_subject[1],
        'Strongest family': strongest_family[0],
        'Strongest family grade': strongest_family[1],
    }


def aggregate_progression_results(consensus, actual_g12=None, tolerance=PROGRESSION_TOLERANCE):
    """Aggregate modeled assessment and observed status without hiding uncertainty."""
    result = consensus.copy()
    if actual_g12 is not None:
        result['Actual G12'] = np.asarray(actual_g12)
        result['Actual Gain'] = result['Actual G12'] - result['external_mean']
        result['Actual Status'] = np.select(
            [
                result['Actual Gain'] >= tolerance,
                result['Actual Gain'] <= -tolerance,
            ],
            ['Improved', 'Declined'],
            default='Maintained',
        )

    def counts(column):
        table = result[column].value_counts().rename_axis('Status').reset_index(name='Count')
        table['Percent'] = table['Count'] / len(result) * 100
        return table

    assessment = counts('Assessment')
    observed = counts('Actual Status') if 'Actual Status' in result else pd.DataFrame()
    sensitivity_rows = []
    for candidate in (0.5, 1.0, 2.0):
        gain = result['Predicted Gain']
        status = np.select(
            [gain >= candidate, gain <= -candidate],
            ['Improved', 'Declined'], default='Maintained'
        )
        sensitivity_rows.append({
            'Tolerance': candidate,
            'Improved': int((status == 'Improved').sum()),
            'Maintained': int((status == 'Maintained').sum()),
            'Declined': int((status == 'Declined').sum()),
            'Uncertain': int((result['Prediction Spread'] > candidate).sum()),
        })
    return {
        'assessment': assessment,
        'observed': observed,
        'sensitivity': pd.DataFrame(sensitivity_rows),
        'complete_cases': len(result),
        'uncertain_count': int((result['Assessment'] == 'Uncertain').sum()),
        'agreement_rate': float((result['Prediction Spread'] <= tolerance).mean() * 100),
    }


def evaluate_progression_models(data, tolerance=PROGRESSION_TOLERANCE):
    """Evaluate direct gain predictions and improved/maintained/declined status."""
    feature_columns = _granular_feature_columns(data)
    numeric_features = [
        column for column in data[feature_columns].select_dtypes(include='number').columns
        if data[column].notna().any()
    ]
    required = ['student sis', 'external_mean', 'target_g12_overall_mean', 'target_g12_gain']
    clean = data[required + [column for column in numeric_features if column not in required]].dropna(
        subset=required
    ).reset_index(drop=True)
    if len(clean) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 'Fewer than 10 complete progression cases.'

    X = clean[numeric_features].replace([np.inf, -np.inf], np.nan)
    y = clean['target_g12_gain']
    model_factories = {
        'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Elastic Net': lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.15, l1_ratio=0.5, max_iter=10000)),
        'PLS': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=min(2, X.shape[1]))),
        'Extra Trees': lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=1),
    }
    predictions = {
        'student sis': clean['student sis'].to_numpy(),
        'External Mean': clean['external_mean'].to_numpy(),
        'Actual G12': clean['target_g12_overall_mean'].to_numpy(),
        'Actual Gain': y.to_numpy(),
    }
    model_names = ['Mean Gain Baseline', *model_factories]
    for name in model_names:
        predictions[name] = np.full(len(clean), np.nan)

    for test_index in range(len(clean)):
        train_indices = np.delete(np.arange(len(clean)), test_index)
        predictions['Mean Gain Baseline'][test_index] = y.iloc[train_indices].mean()
        train_X = X.iloc[train_indices].copy()
        test_X = X.iloc[[test_index]].copy()
        train_medians = train_X.median().fillna(y.iloc[train_indices].mean())
        train_X = train_X.fillna(train_medians)
        test_X = test_X.fillna(train_medians)
        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_X, y.iloc[train_indices])
            predictions[name][test_index] = float(
                np.asarray(model.predict(test_X)).ravel()[0]
            )

    prediction_frame = pd.DataFrame(predictions)
    metric_rows = []
    for name in model_names:
        actual = prediction_frame['Actual Gain']
        predicted = prediction_frame[name]
        metric_rows.append({
            'Model': name,
            'N': len(actual),
            'Gain MAE': mean_absolute_error(actual, predicted),
            'Gain RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'Gain R2': r2_score(actual, predicted),
            'Gain Spearman rho': spearmanr(actual, predicted).statistic,
        })
    metrics = pd.DataFrame(metric_rows).sort_values('Gain MAE').reset_index(drop=True)

    status_rows = []
    actual_status = np.where(
        prediction_frame['Actual Gain'] >= tolerance, 'Improved',
        np.where(prediction_frame['Actual Gain'] <= -tolerance, 'Declined', 'Maintained')
    )
    for name in model_names:
        predicted_status = np.where(
            prediction_frame[name] >= tolerance, 'Improved',
            np.where(prediction_frame[name] <= -tolerance, 'Declined', 'Maintained')
        )
        status_rows.append({
            'Model': name,
            'Tolerance': tolerance,
            'Accuracy': accuracy_score(actual_status, predicted_status),
            'Macro F1': f1_score(actual_status, predicted_status, average='macro', zero_division=0),
        })
    status_metrics = pd.DataFrame(status_rows).sort_values('Macro F1', ascending=False).reset_index(drop=True)
    return metrics, prediction_frame, status_metrics, None
