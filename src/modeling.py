import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RANDOM_STATE = 42


def train_all_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Classifier": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def cross_validate_models(models, X_train, y_train, cv=5):
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores,
        }
    return cv_results


def save_best_model(models, X_test, y_test):
    best_name = None
    best_score = -1
    best_model = None

    for name, model in models.items():
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    return best_name, best_score
