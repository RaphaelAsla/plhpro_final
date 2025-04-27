import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score


class KNN:
    def __init__(self, response_column="Ανταπόκριση", test_size=0.2, random_state=42):
        self.response_column = response_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.best_k = None
        self.results = []
        self.detailed_results = []
        self.validation_metrics = None
        self.class_precision_scores = {}

    def fit(self, train_data, k_range=range(2, 21), fold_range=range(2, 11)):
        print("Preparing data...")
        X = train_data.drop(self.response_column, axis=1)
        y = train_data[self.response_column]

        y = self.label_encoder.fit_transform(y)

        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        print("\nSplitting data into train and validation sets...")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        knn = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", KNeighborsClassifier()),
            ]
        )

        param_grid = {"classifier__n_neighbors": k_range}

        print("Starting GridSearch across folds...\n")
        for c in fold_range:
            print(f"Fold {c}: Running GridSearchCV...")
            grid_search = GridSearchCV(
                knn,
                param_grid,
                cv=StratifiedKFold(n_splits=c),
                scoring={"precision": "precision_macro", "accuracy": "accuracy"},
                refit="precision",
                # refit="accuracy",
                n_jobs=-1,
                return_train_score=True,
            )

            grid_search.fit(X_train, y_train)

            best_k = grid_search.best_params_["classifier__n_neighbors"]

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_valid)
            val_acc = accuracy_score(y_valid, y_pred)
            val_prec = precision_score(
                y_valid, y_pred, average="macro", zero_division=0
            )

            self.results.append(
                {
                    "cv": c,
                    "best_k": best_k,
                    "val_accuracy": val_acc,
                    "val_precision": val_prec,
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
                        "k": k,
                        "cv_precision": mean_precision,
                        "cv_accuracy": mean_accuracy,
                    }
                )

            print(f"Fold {c} completed! Best k = {best_k}\n")

        results_df = pd.DataFrame(self.results)
        self.best_k = int(results_df["best_k"].mode().iloc[0])

        print(f"\nUsing k = {self.best_k} (most common best k among folds)")

        self.final_model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=self.best_k)),
            ]
        )

        print("\nTraining final model on validation data...")
        self.final_model.fit(X_train, y_train)

        validation_predictions = self.final_model.predict(X_valid)
        validation_accuracy = accuracy_score(y_valid, validation_predictions)
        validation_precision = precision_score(
            y_valid, validation_predictions, average="macro", zero_division=0
        )

        self.validation_metrics = {
            "accuracy": validation_accuracy,
            "precision": validation_precision,
        }

        print("\nFinal Validation Metrics:")
        print(f"  • Validation Accuracy: {validation_accuracy:.4f}")
        print(f"  • Validation Precision (macro): {validation_precision:.4f}")

        print("\nClass-specific precision scores:")
        unique_classes = self.label_encoder.classes_
        for i, class_name in enumerate(unique_classes):
            y_true_binary = (y_valid == i).astype(int)
            y_pred_binary = (validation_predictions == i).astype(int)
            class_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            self.class_precision_scores[class_name] = class_prec
            print(f"  - {class_name}: {class_prec:.4f}")

        print("\nTraining final model on full data...")
        self.final_model.fit(X, y)

        return self

    def predict(self, new_data, output_path=None):
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        print("\nPredicting on new campaign data...")
        predictions_new = self.final_model.predict(new_data)
        predictions_new = self.label_encoder.inverse_transform(predictions_new)

        result_df = new_data.copy()
        result_df[self.response_column] = predictions_new

        if output_path:
            result_df.to_excel(output_path, index=False)
            print(f"\nPredictions saved to {output_path}.")

        return result_df

    def get_metrics(self):
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        return {
            "best_k_per_fold": self.results,
            "all_k_per_fold": self.detailed_results,
            "validation_metrics": self.validation_metrics,
            "per_class_metrics": self.class_precision_scores,
        }
