from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import joblib
import traceback
import uuid
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import scipy.stats as stats

warnings.filterwarnings('ignore')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
VECTORIZER_FOLDER = "vectorizers"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(VECTORIZER_FOLDER, exist_ok=True)

MAX_CATEGORIES = 50
MAX_FEATURES = 1000


def clear_old_files():
    for folder in [MODEL_FOLDER, VECTORIZER_FOLDER]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))


def read_csv_with_encoding(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252',
                 'cp850', 'iso-8859-15', 'mac_roman', 'utf-16', 'utf-32']
    df = None
    df_raw = None
    used_encoding = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            df_raw = pd.read_csv(filepath, encoding=enc, dtype=str, keep_default_na=False)
            if len(df.columns) > 0:
                used_encoding = enc
                break
        except:
            continue
    if df is None:
        try:
            df = pd.read_csv(filepath, encoding='utf-8', errors='replace')
            df_raw = pd.read_csv(filepath, encoding='utf-8', errors='replace', dtype=str, keep_default_na=False)
            used_encoding = 'utf-8-replace'
        except:
            pass
    if df is None or len(df.columns) == 0:
        raise Exception("Could not read CSV. File may be corrupted or not a valid CSV.")
    return df, df_raw, used_encoding


def auto_select_target(data):
    cols = data.columns.tolist()
    target_names = ['target', 'label', 'class', 'y', 'outcome', 'result',
                    'prediction', 'predict', 'category', 'type', 'status',
                    'grade', 'score', 'rating', 'rank', 'decision', 'flag',
                    'spam', 'ham', 'sentiment', 'class_label']
    for col in cols:
        if col.lower() in target_names or any(t in col.lower() for t in target_names):
            if data[col].nunique() <= 100 and data[col].nunique() > 1:
                return col
    for col in reversed(cols[-3:]):
        if data[col].nunique() <= 100 and data[col].nunique() > 1:
            return col
    for col in cols:
        if data[col].nunique() == 2:
            return col
    return cols[-1]


def auto_select_features(data, target_col):
    cols = data.columns.tolist()
    features = []
    for col in cols:
        if col == target_col:
            continue
        unique_count = data[col].nunique()
        missing_pct = data[col].isna().sum() / len(data) * 100
        if missing_pct > 80 or unique_count <= 1 or unique_count == len(data):
            continue
        if data[col].dtype == 'object' and unique_count > 500:
            continue
        features.append(col)
    return features


def infer_column_type(series):
    if series.dtype == 'object':
        converted = pd.to_numeric(series, errors='coerce')
        if converted.notna().sum() / len(series) > 0.8:
            return 'numeric'
        try:
            if pd.to_datetime(series, errors='coerce').notna().sum() / len(series) > 0.8:
                return 'datetime'
        except:
            pass
        return 'categorical'
    elif np.issubdtype(series.dtype, np.number):
        return 'numeric'
    elif np.issubdtype(series.dtype, np.datetime64):
        return 'datetime'
    return 'categorical'


def convert_to_numeric(series, col_name):
    if series.dtype == 'object':
        cleaned = series.astype(str).str.replace(r'[$,€£¥%]', '', regex=True)
        cleaned = cleaned.str.replace(r'\s+', '', regex=True)
        converted = pd.to_numeric(cleaned, errors='coerce')
        if converted.notna().sum() / len(series) > 0.5:
            return converted
    return pd.to_numeric(series, errors='coerce')


def robust_clean_data(X, y, problem_type):
    cleaning_log = []
    X = X.copy()
    y = y.copy()
    final_features = pd.DataFrame(index=X.index)
    encoders = {}
    for col in X.columns:
        col_type = infer_column_type(X[col])
        if col_type == 'numeric':
            converted = convert_to_numeric(X[col], col)
            if converted.notna().sum() == 0:
                cleaning_log.append(f"'{col}': Could not convert to numeric, skipped")
                continue
            final_features[col] = converted
            final_features[col] = final_features[col].fillna(converted.median())
        elif col_type == 'categorical':
            unique_count = X[col].nunique()
            if unique_count <= 2:
                le = LabelEncoder()
                final_features[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))
                encoders[col] = le
                cleaning_log.append(f"'{col}': Binary categorical encoded (2 categories)")
            elif unique_count <= MAX_CATEGORIES:
                dummies = pd.get_dummies(X[col].astype(str).fillna("Missing"), prefix=col, drop_first=True)
                for dummy_col in dummies.columns:
                    final_features[dummy_col] = dummies[dummy_col].values
                cleaning_log.append(f"'{col}': One-hot encoded ({unique_count} categories)")
            else:
                freq_map = X[col].value_counts().to_dict()
                final_features[col] = X[col].map(freq_map).fillna(0)
                cleaning_log.append(f"'{col}': Frequency encoded ({unique_count} unique values)")
        elif col_type == 'datetime':
            dt_series = pd.to_datetime(X[col], errors='coerce')
            final_features[f'{col}_year'] = dt_series.dt.year.fillna(dt_series.dt.year.median())
            final_features[f'{col}_month'] = dt_series.dt.month.fillna(dt_series.dt.month.median())
            final_features[f'{col}_day'] = dt_series.dt.day.fillna(dt_series.dt.day.median())
            cleaning_log.append(f"'{col}': Datetime decomposed to year/month/day")

    y_clean = y.copy()
    if y_clean.isna().any():
        valid_mask = y_clean.notna()
        final_features = final_features[valid_mask]
        y_clean = y_clean[valid_mask]
        cleaning_log.append(f"Removed {(~valid_mask).sum()} rows with missing target")

    target_encoder = None
    if y_clean.dtype == 'object':
        le = LabelEncoder()
        y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)
        target_encoder = le
        cleaning_log.append("Target converted from text to numeric labels")

    if problem_type == "regression":
        numeric_cols = final_features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            original_len = len(final_features)
            z_scores = np.abs(stats.zscore(final_features[numeric_cols].fillna(0)))
            outlier_mask = (z_scores < 3).all(axis=1)
            final_features = final_features[outlier_mask]
            y_clean = y_clean[outlier_mask]
            removed = original_len - len(final_features)
            if removed > 0:
                cleaning_log.append(f"Removed {removed} outlier rows (Z-score > 3)")

    for col in final_features.columns:
        final_features[col] = pd.to_numeric(final_features[col], errors='coerce')
    final_features = final_features.fillna(0)

    cols_to_keep = [col for col in final_features.columns if final_features[col].nunique() > 1]
    if len(cols_to_keep) < final_features.shape[1]:
        removed = final_features.shape[1] - len(cols_to_keep)
        final_features = final_features[cols_to_keep]
        cleaning_log.append(f"Removed {removed} constant/zero columns")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(final_features)
    X_scaled = pd.DataFrame(X_scaled, columns=final_features.columns, index=final_features.index)
    cleaning_log.append(f"Applied StandardScaler to {X_scaled.shape[1]} features")

    pca = None
    if X_scaled.shape[1] > 50:
        n_components = min(50, X_scaled.shape[1], len(X_scaled) - 1)
        pca = PCA(n_components=n_components)
        X_final = pca.fit_transform(X_scaled)
        explained_var = sum(pca.explained_variance_ratio_) * 100
        cleaning_log.append(f"PCA: {X_scaled.shape[1]} -> {n_components} features ({explained_var:.1f}% variance)")
    else:
        X_final = X_scaled.values

    return X_final, y_clean, scaler, pca, encoders, target_encoder, cleaning_log


def check_and_apply_smote(X_train, y_train, problem_type):
    if problem_type != "classification":
        return X_train, y_train, False
    class_counts = pd.Series(y_train).value_counts()
    min_class = class_counts.min()
    max_class = class_counts.max()
    if min_class / max_class < 0.3:
        try:
            if min_class >= 6:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class - 1))
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                return X_resampled, y_resampled, True
        except:
            pass
    return X_train, y_train, False


def preprocess_features_manual(X, max_categories=MAX_CATEGORIES, max_features=MAX_FEATURES):
    skipped_columns = []
    label_encoders = {}
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_numeric = X[numeric_cols].copy()
    X_numeric = X_numeric.fillna(X_numeric.median())
    X_processed = X_numeric.copy()
    for col in categorical_cols:
        unique_count = X[col].nunique()
        if unique_count == 1:
            skipped_columns.append(f"{col} (only 1 unique value)")
            continue
        elif unique_count <= max_categories:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed, dummies], axis=1)
        elif unique_count <= 500:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))
            label_encoders[col] = le
        else:
            skipped_columns.append(f"{col} ({unique_count} unique values - too many)")
            continue
    total_features = X_processed.shape[1]
    if total_features > max_features:
        feature_variances = X_processed.var()
        top_features = feature_variances.nlargest(max_features).index.tolist()
        X_processed = X_processed[top_features]
        skipped_columns.append(f"Reduced from {total_features} to {max_features} features")
    return X_processed, skipped_columns, label_encoders


def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Bagging": BaggingClassifier(n_estimators=100, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
    }


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100),
        "Bagging Regressor": BaggingRegressor(n_estimators=100, n_jobs=-1),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    columns = []
    error_message = None
    results = []
    detected_problem = None
    uploaded_file = None
    is_nlp = False
    problem_type = None
    warnings_list = []
    encoding_used = None
    cleaning_log = []
    auto_features = []
    auto_target = None
    smote_applied = False

    if request.method == "POST":
        try:
            file = request.files.get("file")
            existing_file = request.form.get("existing_file")
            run_mode = request.form.get("run_mode", "step_by_step")

            if file and file.filename:
                ext = os.path.splitext(file.filename)[1]
                unique_name = str(uuid.uuid4())[:8] + ext
                filepath = os.path.join(UPLOAD_FOLDER, unique_name)
                file.save(filepath)
                uploaded_file = unique_name
            elif existing_file:
                filepath = os.path.join(UPLOAD_FOLDER, existing_file)
                if not os.path.exists(filepath):
                    error_message = "Previous upload not found. Please upload again."
                    return render_template("model.html", error=error_message)
                uploaded_file = existing_file
            else:
                error_message = "Please upload a dataset."
                return render_template("model.html", error=error_message)

            data, data_raw, encoding_used = read_csv_with_encoding(filepath)
            columns = data.columns.tolist()

            # ==================== AUTO MODE ====================
            if run_mode == "auto":
                auto_target = auto_select_target(data)
                auto_features = auto_select_features(data, auto_target)

                if not auto_features:
                    error_message = "Could not auto-select features. Dataset may have only ID columns."
                    return render_template("model.html", error=error_message, step="upload")

                X = data[auto_features].copy()
                y = data[auto_target].copy()

                if X.shape[1] == 1 and X.dtypes.iloc[0] == "object":
                    is_nlp = True
                    detected_problem = "NLP Classification"
                    problem_type = "classification"
                elif y.nunique() <= 10:
                    detected_problem = "Classification"
                    problem_type = "classification"
                else:
                    detected_problem = "Regression"
                    problem_type = "regression"

                if is_nlp:
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X_processed = vectorizer.fit_transform(X.iloc[:, 0].astype(str).fillna(""))
                    cleaning_log = ["Applied TF-IDF vectorization for text data"]
                    scaler = None
                    pca = None
                    encoders = {}
                    target_encoder = None
                else:
                    X_processed, y, scaler, pca, encoders, target_encoder, cleaning_log = robust_clean_data(X, y, problem_type)
                    vectorizer = None

                if len(data) < 10:
                    error_message = "Dataset too small. Need at least 10 rows."
                    return render_template("model.html", error=error_message, step="upload")

                test_size = 0.2 if len(data) >= 50 else 0.3
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=42
                )

                if problem_type == "classification":
                    X_train, y_train, smote_applied = check_and_apply_smote(X_train, y_train, problem_type)
                    if smote_applied:
                        cleaning_log.append("Applied SMOTE to balance imbalanced classes")

                if detected_problem in ["Classification", "NLP Classification"]:
                    models = get_classification_models()
                else:
                    models = get_regression_models()

                clear_old_files()

                results = []
                trained_models = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    if detected_problem in ["Classification", "NLP Classification"]:
                        score = accuracy_score(y_test, predictions)
                        performance = round(score * 100, 2)
                        metric = "Accuracy"
                    else:
                        mse = mean_squared_error(y_test, predictions)
                        rmse = round(np.sqrt(mse), 2)
                        performance = round(mse, 2)
                        metric = "MSE"
                    trained_models[name] = {
                        "model": model,
                        "metric": metric,
                        "performance": performance,
                    }
                    results.append({
                        "model": name,
                        "metric": metric,
                        "performance": performance,
                        "accuracy": performance if metric == "Accuracy" else None,
                        "mse": performance if metric == "MSE" else None,
                        "rmse": rmse if metric == "MSE" else None,
                    })

                # Ensemble
                if len(trained_models) >= 2:
                    if detected_problem in ["Classification", "NLP Classification"]:
                        ensemble = VotingClassifier(
                            estimators=[(name, m["model"]) for name, m in trained_models.items()],
                            voting="hard"
                        )
                    else:
                        ensemble = VotingRegressor(
                            estimators=[(name, m["model"]) for name, m in trained_models.items()]
                        )
                    ensemble_name = "Ensemble (Voting)"
                    ensemble.fit(X_train, y_train)
                    ensemble_preds = ensemble.predict(X_test)
                    if detected_problem in ["Classification", "NLP Classification"]:
                        ens_score = accuracy_score(y_test, ensemble_preds)
                        ens_performance = round(ens_score * 100, 2)
                        ens_metric = "Accuracy"
                    else:
                        ens_mse = mean_squared_error(y_test, ensemble_preds)
                        ens_rmse = round(np.sqrt(ens_mse), 2)
                        ens_performance = round(ens_mse, 2)
                        ens_metric = "MSE"
                    trained_models[ensemble_name] = {
                        "model": ensemble,
                        "metric": ens_metric,
                        "performance": ens_performance,
                    }
                    results.append({
                        "model": ensemble_name,
                        "metric": ens_metric,
                        "performance": ens_performance,
                        "accuracy": ens_performance if ens_metric == "Accuracy" else None,
                        "mse": ens_performance if ens_metric == "MSE" else None,
                        "rmse": ens_rmse if ens_metric == "MSE" else None,
                    })

                if detected_problem in ["Classification", "NLP Classification"]:
                    results.sort(key=lambda x: x["performance"], reverse=True)
                else:
                    results.sort(key=lambda x: x["performance"])

                app.config["CURRENT_TRAINED_MODELS"] = trained_models
                app.config["CURRENT_VECTORIZER"] = vectorizer
                app.config["CURRENT_IS_NLP"] = is_nlp
                app.config["CURRENT_SCALER"] = scaler
                app.config["CURRENT_PCA"] = pca
                app.config["CURRENT_TARGET_ENCODER"] = target_encoder

                return render_template(
                    "model.html",
                    columns=columns,
                    results=results,
                    detected_problem=detected_problem,
                    uploaded_file=uploaded_file,
                    step="results",
                    is_nlp=is_nlp,
                    problem_type=problem_type,
                    cleaning_log=cleaning_log,
                    auto_mode=True,
                    auto_target=auto_target,
                    auto_features=auto_features,
                    encoding_used=encoding_used,
                    smote_applied=smote_applied
                )

            # ==================== MANUAL MODE ====================
            else:
                selected_features = request.form.getlist("features")
                target_column = request.form.get("target")
                mode = request.form.get("mode", "auto")

                if not selected_features or not target_column:
                    sample_y = data.iloc[:, -1]
                    if data.shape[1] > 1 and data.iloc[:, :-1].shape[1] == 1 and data.iloc[:, 0].dtype == "object":
                        problem_type = "classification"
                    elif sample_y.nunique() <= 10:
                        problem_type = "classification"
                    else:
                        problem_type = "regression"

                    column_types = {}
                    column_uniques = {}
                    column_samples = {}

                    for col in columns:
                        dtype = str(data[col].dtype)
                        if dtype == 'object':
                            column_types[col] = 'Text'
                        elif 'int' in dtype:
                            column_types[col] = 'Integer'
                        elif 'float' in dtype:
                            column_types[col] = 'Decimal'
                        else:
                            column_types[col] = dtype
                        column_uniques[col] = data[col].nunique()
                        if col in data_raw.columns:
                            raw_vals = data_raw[col].head(3).tolist()
                        else:
                            raw_vals = data[col].astype(str).head(3).tolist()
                        sample_strs = []
                        for v in raw_vals:
                            s = str(v)
                            if len(s) > 60:
                                s = s[:57] + "..."
                            sample_strs.append(s)
                        column_samples[col] = " | ".join(sample_strs) if sample_strs else "(empty)"

                    return render_template(
                        "model.html",
                        columns=columns,
                        uploaded_file=uploaded_file,
                        step="select",
                        problem_type=problem_type,
                        column_types=column_types,
                        column_uniques=column_uniques,
                        column_samples=column_samples
                    )

                if target_column not in data.columns:
                    error_message = f"Target column '{target_column}' not found."
                    return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                         step="select", error=error_message)

                invalid_features = [f for f in selected_features if f not in data.columns]
                if invalid_features:
                    error_message = f"Invalid features: {', '.join(invalid_features)}"
                    return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                         step="select", error=error_message)

                if target_column in selected_features:
                    selected_features = [f for f in selected_features if f != target_column]
                    if not selected_features:
                        error_message = "Target cannot be the only feature."
                        return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                             step="select", error=error_message)

                X = data[selected_features].copy()
                y = data[target_column].copy()

                if y.isna().any():
                    error_message = "Target has missing values. Clean your data."
                    return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                         step="select", error=error_message)

                if X.shape[1] == 1 and X.dtypes.iloc[0] == "object":
                    is_nlp = True
                    detected_problem = "NLP Classification"
                    problem_type = "classification"
                elif y.nunique() <= 10:
                    detected_problem = "Classification"
                    problem_type = "classification"
                else:
                    detected_problem = "Regression"
                    problem_type = "regression"

                vectorizer = None
                if is_nlp:
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X = vectorizer.fit_transform(X.iloc[:, 0].astype(str).fillna(""))
                else:
                    X, skipped_columns, label_encoders = preprocess_features_manual(X)
                    if skipped_columns:
                        warnings_list = skipped_columns
                    if X.shape[1] == 0:
                        error_message = "No valid features after preprocessing."
                        return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                             step="select", error=error_message)

                if len(data) < 10:
                    error_message = "Dataset too small. Need at least 10 rows."
                    return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                         step="select", error=error_message)

                test_size = 0.2 if len(data) >= 50 else 0.3
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                models = {}
                if mode == "auto":
                    if detected_problem in ["Classification", "NLP Classification"]:
                        models = get_classification_models()
                    else:
                        models = get_regression_models()
                else:
                    selected_models = request.form.getlist("models")
                    if not selected_models:
                        error_message = "Please select at least one model in Manual Mode."
                        return render_template("model.html", columns=columns, uploaded_file=uploaded_file,
                                             step="select", error=error_message, problem_type=problem_type)

                    for model_name in selected_models:
                        if model_name == "Logistic Regression":
                            models[model_name] = LogisticRegression(max_iter=1000)
                        elif model_name == "Random Forest":
                            models[model_name] = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                        elif model_name == "Gradient Boosting":
                            models[model_name] = GradientBoostingClassifier(n_estimators=100)
                        elif model_name == "AdaBoost":
                            models[model_name] = AdaBoostClassifier(n_estimators=100)
                        elif model_name == "Bagging":
                            models[model_name] = BaggingClassifier(n_estimators=100, n_jobs=-1)
                        elif model_name == "Extra Trees":
                            models[model_name] = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
                        elif model_name == "Linear Regression":
                            models[model_name] = LinearRegression()
                        elif model_name == "Random Forest Regressor":
                            models[model_name] = RandomForestRegressor(n_estimators=100, n_jobs=-1)
                        elif model_name == "Gradient Boosting Regressor":
                            models[model_name] = GradientBoostingRegressor(n_estimators=100)
                        elif model_name == "AdaBoost Regressor":
                            models[model_name] = AdaBoostRegressor(n_estimators=100)
                        elif model_name == "Bagging Regressor":
                            models[model_name] = BaggingRegressor(n_estimators=100, n_jobs=-1)
                        elif model_name == "Extra Trees Regressor":
                            models[model_name] = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)

                clear_old_files()

                results = []
                trained_models = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    if detected_problem in ["Classification", "NLP Classification"]:
                        score = accuracy_score(y_test, predictions)
                        performance = round(score * 100, 2)
                        metric = "Accuracy"
                    else:
                        mse = mean_squared_error(y_test, predictions)
                        rmse = round(np.sqrt(mse), 2)
                        performance = round(mse, 2)
                        metric = "MSE"
                    trained_models[name] = {
                        "model": model,
                        "metric": metric,
                        "performance": performance,
                    }
                    results.append({
                        "model": name,
                        "metric": metric,
                        "performance": performance,
                        "accuracy": performance if metric == "Accuracy" else None,
                        "mse": performance if metric == "MSE" else None,
                        "rmse": rmse if metric == "MSE" else None,
                    })

                # Ensemble
                if len(trained_models) >= 2:
                    if detected_problem in ["Classification", "NLP Classification"]:
                        ensemble = VotingClassifier(
                            estimators=[(name, m["model"]) for name, m in trained_models.items()],
                            voting="hard"
                        )
                    else:
                        ensemble = VotingRegressor(
                            estimators=[(name, m["model"]) for name, m in trained_models.items()]
                        )
                    ensemble_name = "Ensemble (Voting)"
                    ensemble.fit(X_train, y_train)
                    ensemble_preds = ensemble.predict(X_test)
                    if detected_problem in ["Classification", "NLP Classification"]:
                        ens_score = accuracy_score(y_test, ensemble_preds)
                        ens_performance = round(ens_score * 100, 2)
                        ens_metric = "Accuracy"
                    else:
                        ens_mse = mean_squared_error(y_test, ensemble_preds)
                        ens_rmse = round(np.sqrt(ens_mse), 2)
                        ens_performance = round(ens_mse, 2)
                        ens_metric = "MSE"
                    trained_models[ensemble_name] = {
                        "model": ensemble,
                        "metric": ens_metric,
                        "performance": ens_performance,
                    }
                    results.append({
                        "model": ensemble_name,
                        "metric": ens_metric,
                        "performance": ens_performance,
                        "accuracy": ens_performance if ens_metric == "Accuracy" else None,
                        "mse": ens_performance if ens_metric == "MSE" else None,
                        "rmse": ens_rmse if ens_metric == "MSE" else None,
                    })

                if detected_problem in ["Classification", "NLP Classification"]:
                    results.sort(key=lambda x: x["performance"], reverse=True)
                else:
                    results.sort(key=lambda x: x["performance"])

                app.config["CURRENT_TRAINED_MODELS"] = trained_models
                app.config["CURRENT_VECTORIZER"] = vectorizer
                app.config["CURRENT_IS_NLP"] = is_nlp

                return render_template(
                    "model.html",
                    columns=columns,
                    results=results,
                    detected_problem=detected_problem,
                    uploaded_file=uploaded_file,
                    step="results",
                    is_nlp=is_nlp,
                    problem_type=problem_type,
                    warnings=warnings_list,
                    encoding_used=encoding_used,
                    cleaning_log=cleaning_log,
                    smote_applied=smote_applied
                )

        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            print(traceback.format_exc())

    return render_template(
        "model.html",
        columns=columns,
        error=error_message,
        step="upload"
    )


@app.route("/download_model/<model_name>")
def download_model(model_name):
    trained_models = app.config.get("CURRENT_TRAINED_MODELS", {})
    if model_name not in trained_models:
        return "Model not found. Please train models first.", 404
    model_data = trained_models[model_name]
    model = model_data["model"]
    timestamp = str(uuid.uuid4())[:6]
    filename = model_name.replace(" ", "_") + "_" + timestamp + ".pkl"
    model_path = os.path.join(MODEL_FOLDER, filename)
    joblib.dump(model, model_path)
    return send_file(model_path, as_attachment=True)


@app.route("/download_vectorizer")
def download_vectorizer():
    vectorizer = app.config.get("CURRENT_VECTORIZER")
    if vectorizer is None:
        return "No vectorizer available.", 404
    timestamp = str(uuid.uuid4())[:6]
    filename = "vectorizer_" + timestamp + ".pkl"
    vectorizer_path = os.path.join(VECTORIZER_FOLDER, filename)
    joblib.dump(vectorizer, vectorizer_path)
    return send_file(vectorizer_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)