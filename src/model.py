import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report


class KNN:
    def __init__(self, k=None, test_size=0.2, random_state=42):
        self.response_column = "Ανταπόκριση"
        self.test_size = test_size
        self.random_state = random_state
        self.best_k = k
        self.results = []
        self.detailed_results = []
        self.X = None
        self.Y = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.preprocessor = None
        self.validation_metrics = None
        self.validation_metrics_str = ""
        self.cv_validation_metrics = None
        self.overall_validation_metrics = None
        self.final_model = None

    def find_best_k(self, k_range=range(2, 21), fold_range=range(2, 11)):
        if self.best_k is not None:
            raise ValueError("k value has already been defined.")
        if self.X is None:
            raise ValueError("Training data has not been fed to network yet. Call feed_data() first.")

        knn = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", KNeighborsClassifier()),
            ]
        )

        param_grid = {"classifier__n_neighbors": k_range}

        for c in fold_range:
            grid_search = GridSearchCV(
                knn,
                param_grid,
                cv=StratifiedKFold(n_splits=c),
                scoring={"precision": "precision_macro", "accuracy": "accuracy"},
                refit="precision", # type: ignore
                n_jobs=-1,
                return_train_score=True,
            )

            grid_search.fit(self.X_train, self.y_train)

            best_k = grid_search.best_params_["classifier__n_neighbors"]

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_valid)
            accuracy = accuracy_score(self.y_valid, y_pred)
            precision = precision_score(self.y_valid, y_pred, average="macro", zero_division=0) # type: ignore

            self.results.append(
                {
                    "cv": c,
                    "best_k": best_k,
                    "accuracy": accuracy,
                    "precision": precision,
                }
            )

            for mean_precision, mean_accuracy, k in zip(
                grid_search.cv_results_["mean_test_precision"],
                grid_search.cv_results_["mean_test_accuracy"],
                grid_search.cv_results_["param_classifier__n_neighbors"],
            ):
                self.detailed_results.append(
                    {
                        "cv": c,
                        "k": int(k),
                        "cv_precision": float(mean_precision),
                        "cv_accuracy": float(mean_accuracy),
                    }
                )

        self.cv_validation_metrics = {
            "best_k_per_fold": pd.DataFrame(self.results),
            "all_k_per_fold": pd.DataFrame(self.detailed_results)
        }

        results_df = pd.DataFrame(self.results)
        self.best_k = int(results_df["best_k"].mode().iloc[0])

    def feed_data(self, train_data):
        self.X = train_data.drop(self.response_column, axis=1)
        self.y = train_data[self.response_column]

        categorical_cols = self.X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = self.X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, test_size=self.test_size, shuffle=True, random_state=self.random_state, stratify=self.y
        )

    def fit(self):
        if self.best_k is None:
            raise ValueError("k value has not been set yet. Call find_best_k() or define it manually when creating the KNN")
        if self.X is None:
            raise ValueError("Training data has not been fed to network yet. Call feed_data() first.")

        self.final_model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=self.best_k)),
            ]
        )

        self.final_model.fit(self.X, self.y)

    def predict(self, new_data, output_path=None):
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        predictions_new = self.final_model.predict(new_data)

        result_df = new_data.copy()
        result_df[self.response_column] = predictions_new

        if output_path:
            result_df.to_excel(output_path, index=False)

        return result_df

    def gen_metrics(self):
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        self.final_model.fit(self.X_train, self.y_train)

        y_pred = self.final_model.predict(self.X_valid)
        report = classification_report(
            self.y_valid, 
            y_pred,
            output_dict=True
        )

        self.validation_metrics = {
            "Accuracy": report['accuracy'], # type: ignore
            "Precision": report['macro avg']['precision'], # type: ignore
            "Yes Precision": report['yes']['recall'], # type: ignore
            "No Precision": report['no']['recall'], # type: ignore
            "Yes Accuracy": report['yes']['precision'], # type: ignore
            "No Accuracy": report['no']['precision'], # type: ignore
            'confusion_matrix': confusion_matrix(self.y_valid, y_pred), # type: ignore
        }

        self.overall_validation_metrics = {
            'cv_validation_metrics': self.cv_validation_metrics,
            'validation_metrics': self.validation_metrics,
            'best_k': self.best_k
        }
        
        if self.validation_metrics:
            self.validation_metrics_str += "\nFinal Validation Metrics:\n"
            self.validation_metrics_str += f"  • Validation Accuracy: {self.validation_metrics['Accuracy']:.4f}\n"
            self.validation_metrics_str += f"  • Validation Precision (macro): {self.validation_metrics['Precision']:.4f}\n"
            self.validation_metrics_str += "\n  • Class-specific Accuracy Scores:\n"
            self.validation_metrics_str += f"    - Yes Accuracy: {self.validation_metrics['Yes Accuracy']:.4f}\n"
            self.validation_metrics_str += f"    - No Accuracy: {self.validation_metrics['No Accuracy']:.4f}\n\n"
            self.validation_metrics_str += f"    - Yes Precision (macro): {self.validation_metrics['Yes Precision']:.4f}\n"
            self.validation_metrics_str += f"    - No Precision (macro): {self.validation_metrics['No Precision']:.4f}\n"
