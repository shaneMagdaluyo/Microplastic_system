
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE


# =========================
# LOAD DATA
# =========================
def load_data(file):
    df = pd.read_csv(file)
    return df


# =========================
# PREPROCESSING
# =========================
def preprocess(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    return preprocessor


# =========================
# MODELS
# =========================
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC()
    }


# =========================
# TRAIN FUNCTION
# =========================
def train_models(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # preprocessing
    preprocessor = preprocess(X)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)

    models = get_models()

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test_enc)

        acc = accuracy_score(y_test, preds)

        results[name] = {
            "accuracy": acc,
            "report": classification_report(y_test, preds)
        }

        if acc > best_score:
            best_score = acc
            best_model = (name, model)

    return results, best_model
