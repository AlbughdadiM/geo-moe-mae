import gc
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


class LogisticRegressionModel:
    def __init__(
        self,
        scale=False,
        max_iter=1000,
        seed=0,
        multilabel=False,
        n_jobs=None,
        verbose=False,
    ):
        self.scale = scale
        self.max_iter = max_iter
        self.seed = seed
        self.multilabel = multilabel
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.scaler = None
        self.model = None

    def fit(self, x_train, y_train):
        # scaling if requested
        if self.scale:
            self.scaler = StandardScaler()
            x_train = self.scaler.fit_transform(x_train)

        base_model = LogisticRegression(
            max_iter=self.max_iter,
            solver="sag",
            random_state=self.seed,
            verbose=1 if self.verbose else 0,
            n_jobs=self.n_jobs,
        )

        if self.multilabel:
            self.model = OneVsRestClassifier(base_model, n_jobs=self.n_jobs)
        else:
            self.model = base_model

        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        if self.scaler is not None:
            x_test = self.scaler.transform(x_test)
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        if self.scaler is not None:
            x_test = self.scaler.transform(x_test)

        y_pred = self.model.predict(x_test)

        metrics = {
            "overall_accuracy": accuracy_score(y_test, y_pred),
            "overall_precision": precision_score(
                y_test, y_pred, average="micro", zero_division=0
            ),
            "average_precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "overall_recall": recall_score(
                y_test, y_pred, average="micro", zero_division=0
            ),
            "average_recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "overall_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "average_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # multilabel-specific metrics
        if self.multilabel:
            score = self.model.predict_proba(x_test)
            metrics.update(
                {
                    "overall_map": average_precision_score(
                        y_test, score, average="micro"
                    ),
                    "average_map": average_precision_score(
                        y_test, score, average="macro"
                    ),
                }
            )

        torch.cuda.empty_cache()
        gc.collect()
        return metrics, y_pred
