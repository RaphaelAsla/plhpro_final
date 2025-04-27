import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Plotter:
    def __init__(self, metrics):
        self.cv_validation_metrics = metrics['cv_validation_metrics']
        self.validation_metrics = metrics["validation_metrics"]
        self.best_k = metrics['best_k']

    def plot_best_cv_metrics_vs_folds(self):
        if self.cv_validation_metrics is None:
            print("This plot is not available, k was set manually.")
            return None

        best_k_per_fold = pd.DataFrame(self.cv_validation_metrics["best_k_per_fold"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            best_k_per_fold["cv"],
            best_k_per_fold["accuracy"],
            marker="o",
            label="Validation Accuracy",
        )
        ax.plot(
            best_k_per_fold["cv"],
            best_k_per_fold["precision"],
            marker="s",
            label="Validation Precision",
        )
        ax.set_xlabel("Number of Folds (best k)")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics vs. Number of Folds (best k)")
        ax.grid(True)
        ax.legend()

        cv_values = best_k_per_fold["cv"]
        best_k_values = best_k_per_fold["best_k"]
        custom_labels = [f"{cv}\n(k={k})" for cv, k in zip(cv_values, best_k_values)]

        ax.set_xticks(cv_values)
        ax.set_xticklabels(custom_labels)
        return fig

    def plot_cv_metrics_vs_k(self, score):
        if self.cv_validation_metrics is None:
            print("This plot is not available, k was set manually.")
            return

        all_k_per_fold = pd.DataFrame(self.cv_validation_metrics["all_k_per_fold"])
        all_k_per_fold["cv"] = all_k_per_fold["cv"].astype("category")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=all_k_per_fold,
            x="k",
            y="cv_"+score,
            hue="cv",
            marker="o",
            ax=ax,
        )
        ax.axvline(
            x=self.best_k, color="red", linestyle="--", label=f"Best k={self.best_k}"
        )
        ax.set_xlabel("k Value")
        ax.set_ylabel("CV Accuracy")
        ax.set_title("CV Accuracy vs. k Value for Different Folds")
        ax.grid(True)
        ax.legend(title="Folds", loc="upper right")

        plt.tight_layout()
        return fig

    def plot_test_validation_metrics(self):
        metrics_order = [
            "Accuracy",
            "Precision",
            "Yes Precision",
            "Yes Accuracy",
            "No Precision",
            "No Accuracy"
        ]
        
        data = {
            "Group": ["Overall", "Overall", "Yes", "Yes", "No", "No"],
            "Metric Type": ["Accuracy", "Precision", "Precision", "Accuracy", "Precision", "Accuracy"],
            "Score": [self.validation_metrics[m] for m in metrics_order]
        }
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="Group",
            y="Score",
            hue="Metric Type",
            palette={"Accuracy": "#1f77b4", "Precision": "#ff7f0e"},
            edgecolor="black"
        )

        plt.ylim(0, 1.1)
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("Metric", fontsize=14)
        plt.title(f"Validation Metrics (k={self.best_k})", fontsize=16, fontweight="bold")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Metric Type", loc="lower right", fontsize=12, title_fontsize=13)
        plt.tight_layout()

        return plt.gcf()

    def plot_cv_metrics_heatmap(self, score):
        if self.cv_validation_metrics is None:
            print("This plot is not available, k was set manually.")
            return None
        
        all_k_per_fold = pd.DataFrame(self.cv_validation_metrics["all_k_per_fold"])
        pivot_df = all_k_per_fold.pivot(index='cv', columns='k', values="cv_"+score)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt=".3f", 
            cmap="YlGnBu", 
            cbar_kws={'label': 'CV Accuracy'}
        )
        
        for i in range(len(pivot_df.index)):
            col_idx = list(pivot_df.columns).index(self.best_k)
            ax.add_patch(plt.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2)) # type: ignore
        
        ax.set_title('CV Accuracy Heatmap: Folds vs. k Values', fontsize=14)
        ax.set_xlabel('k Value', fontsize=12)
        ax.set_ylabel('Number of CV Folds', fontsize=12)
        
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self):
        class_names = ["No", "Yes"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            self.validation_metrics["confusion_matrix"], 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            annot_kws={"size": 16}
        )
        
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)
        ax.set_title(f'Confusion Matrix (k={self.best_k})', fontsize=16)
        
        plt.tight_layout()
        return fig
