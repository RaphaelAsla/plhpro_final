import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score

#!!! ΔΕΝ ΤΟ ΕΧΩ "ΣΥΝΔΕΣΕΙ" ΑΚΟΜΗ ΜΕ ΤΟ UI !!!"

print("Loading data...")
train_data = pd.read_excel("./data/Project40PastCampaignData.xlsx")
new_data = pd.read_excel("./data/Project40NewCampaignData.xlsx")

X = train_data.drop("Ανταπόκριση", axis=1)
y = train_data["Ανταπόκριση"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

print("\nSplitting data into train and validation sets...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier()),
    ]
)

k_values = range(2, 21)  # Αρχιζω απο 2 για να αποτρεψω overfitting
num_folds = range(2, 11)

param_grid = {"classifier__n_neighbors": k_values}

results = []
detailed_results = []

print("Starting GridSearch across folds...\n")
for c in num_folds:
    print(f"Fold {c}: Running GridSearchCV...")
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=StratifiedKFold(n_splits=c),
        # scoring='accuracy', # Δεν φαινεται να εχει διαφορα στην τελικη τιμη του k
        scoring="precision_macro",
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    best_k = grid_search.best_params_["classifier__n_neighbors"]
    best_cv_score = grid_search.best_score_

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_valid)
    val_acc = accuracy_score(y_valid, y_pred)
    val_prec = precision_score(y_valid, y_pred, average="macro", zero_division=0)

    results.append(
        {
            "cv": c,
            "best_k": best_k,
            "cv_precision": best_cv_score,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
        }
    )

    for mean_score, k in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["param_classifier__n_neighbors"],
    ):
        detailed_results.append({"cv": c, "k": int(k), "cv_precision": mean_score})

    print(f"Fold {c} completed! Best k = {best_k}\n")

results_df = pd.DataFrame(results)
detailed_results_df = pd.DataFrame(detailed_results)

best_k_value = int(results_df["best_k"].mode().iloc[0])

print(f"\nUsing k = {best_k_value} (most common best k among folds)")

final_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k_value)),
    ]
)

print("\nTraining final model on full data...")
final_model.fit(X, y)

validation_predictions = final_model.predict(X_valid)
validation_accuracy = accuracy_score(y_valid, validation_predictions)
validation_precision = precision_score(
    y_valid, validation_predictions, average="macro", zero_division=0
)

print("\nFinal Validation Metrics:")
print(f"  • Validation Accuracy: {validation_accuracy:.4f}")
print(f"  • Validation Precision (macro): {validation_precision:.4f}")

print("\nClass-specific precision scores:")
unique_classes = label_encoder.classes_
for i, class_name in enumerate(unique_classes):
    y_true_binary = (y_valid == i).astype(int)
    y_pred_binary = (validation_predictions == i).astype(int)
    class_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    print(f"  - {class_name}: {class_prec:.4f}")

print("\nPredicting on new campaign data...")
predictions_new = final_model.predict(new_data)
predictions_new = label_encoder.inverse_transform(predictions_new)
new_data["Ανταπόκριση"] = predictions_new
new_data.to_excel("predictions.xlsx", index=False)

print("\nPredictions saved to 'predictions.xlsx'.")

# plotting

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="k",
    y="cv_precision",
    hue="cv",
    data=detailed_results_df,
    palette="tab10",
    marker="o",
)
plt.title("Precision vs k for each Fold")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Precision (Macro)")
plt.legend(title="Folds", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()
