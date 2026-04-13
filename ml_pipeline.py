import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report


def load_data(file):
    return pd.read_csv(file)


def clean_data(df, target):
    df = df.copy()

    y = df[target]
    X = df.drop(columns=[target])

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = pd.DataFrame(X)

    return X, y


def train_models(df, target):
    X, y = clean_data(df, target)
    y = pd.Series(y)

    if y.nunique() < 2:
        raise ValueError("Target must have at least 2 classes")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    best_model = None
    best_name = ""
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results[name] = {
            "accuracy": acc,
            "report": classification_report(y_test, preds)
        }

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    return results, best_name, best_model


def save_model(model):
    joblib.dump(model, "best_model.pkl")
