import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score

#!!! ΔΕΝ ΤΟ ΕΧΩ "ΣΥΝΔΕΣΕΙ" ΑΚΟΜΗ ΜΕ ΤΟ UI !!!"
#!!! τρεχει κανονικα αλλα δεν ειναι 100% ετοιμο, θελω να δω καποια πραγματα ακομη που μου φαινονται λιγο περιεργα ↵ !!!
#!!! (οταν χρησιμοποιω accuracy_score για να βρω το καλυτερο k, μου βγαζει k = 1, (overfitting) ακομη δεν εχω καταλαβει γιατι), αλλα κατα τάλλα ειναι οκ, θα κανω και μερικα plots

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
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier()),
    ]
)

param_grid = {"classifier__n_neighbors": range(1, 21)}
results = []

for c in range(2, 6):
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=StratifiedKFold(n_splits=c),
        #scoring='accuracy',
        scoring='precision_macro',
        n_jobs=-1,
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
    print(f"Currently on {c} folds, with best k = {best_k}")

results_df = pd.DataFrame(results)

best_k_value = int(results_df["best_k"].mode().iloc[0])

print(f"\nUsing k = {best_k_value} as it is the most common among the different folds")

final_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k_value)),
    ]
)

final_model.fit(X, y)

validation_predictions = final_model.predict(X_valid)
validation_accuracy = accuracy_score(y_valid, validation_predictions)
validation_precision = precision_score(y_valid, validation_predictions, average="macro", zero_division=0)

print(f"\nValidation Accuracy: {validation_accuracy:.4f}")
print(f"Validation Precision (macro): {validation_precision:.4f}")

print("\nClass-specific precision:")
unique_classes = label_encoder.classes_
for i, class_name in enumerate(unique_classes):
    y_true_binary = (y_valid == i).astype(int)
    y_pred_binary = (validation_predictions == i).astype(int)
    class_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    print(f"  {class_name}: {class_prec:.4f}")

predictions_new = final_model.predict(new_data)
predictions_new = label_encoder.inverse_transform(predictions_new)
new_data["Ανταπόκριση"] = predictions_new
new_data.to_excel("predictions.xlsx", index=False)

print("\nPredictions saved to 'predictions.xlsx'")
