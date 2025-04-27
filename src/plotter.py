import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self, metrics):
        self.best_k_per_fold = pd.DataFrame(metrics["best_k_per_fold"])
        self.all_k_per_fold = pd.DataFrame(metrics["all_k_per_fold"])
        self.validation_metrics = metrics["validation_metrics"]
        self.per_class_metrics = metrics["per_class_validation_metrics"]

        self.best_k = None if self.best_k_per_fold.empty else int(self.best_k_per_fold["best_k"].mode().iloc[0])

    def best_validation_metrics_vs_number_of_folds(self):
        if self.best_k is None:
            raise ValueError("This plot is not available, k was set manually.")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            self.best_k_per_fold["cv"],
            self.best_k_per_fold["val_accuracy"],
            marker="o",
            label="Validation Accuracy",
        )
        ax.plot(
            self.best_k_per_fold["cv"],
            self.best_k_per_fold["val_precision"],
            marker="s",
            label="Validation Precision",
        )
        ax.set_xlabel("Number of Folds (best k)")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics vs. Number of Folds (best k)")
        ax.grid(True)
        ax.legend()

        cv_values = self.best_k_per_fold["cv"]
        best_k_values = self.best_k_per_fold["best_k"]
        custom_labels = [f"{cv}\n(k={k})" for cv, k in zip(cv_values, best_k_values)]

        ax.set_xticks(cv_values)
        ax.set_xticklabels(custom_labels)
        return fig

    def cv_precision_vs_k_value_for_different_folds(self):
        if self.best_k is None:
            raise ValueError("This plot is not available, k was set manually.")

        self.all_k_per_fold["cv"] = self.all_k_per_fold["cv"].astype("category")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=self.all_k_per_fold,
            x="k",
            y="cv_precision",
            hue="cv",
            marker="o",
            ax=ax,
        )
        ax.axvline(
            x=self.best_k, color="red", linestyle="--", label=f"Best k={self.best_k}"
        )
        ax.set_xlabel("k Value")
        ax.set_ylabel("CV Precision")
        ax.set_title("CV Precision vs. k Value for Different Folds")
        ax.grid(True)
        ax.legend(title="Folds", loc="upper right")
        return fig

    def cv_accuracy_vs_k_value_for_different_folds(self):
        if self.best_k is None:
            raise ValueError("This plot is not available, k was set manually.")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=self.all_k_per_fold,
            x="k",
            y="cv_accuracy",
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

    def final_validation_metrics(self):
        metrics = ["Accuracy", "Precision"]
        values = [self.validation_metrics[m] for m in metrics]

        class_names = list(self.per_class_metrics.keys())
        class_values = list(self.per_class_metrics.values())

        all_labels = metrics + class_names
        all_values = values + class_values

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        if len(all_labels) > len(colors):
            colors = colors * (len(all_labels) // len(colors) + 1)

        fig, ax = plt.subplots(figsize=(12, 7))

        bars = ax.bar(
            all_labels,
            all_values,
            color=colors[: len(all_labels)],
            edgecolor="black",
            linewidth=1.2,
            width=0.6,
        )

        ax.set_ylim(0, 1.1)
        ax.set_title(
            f"Validation Metrics (k={self.best_k})", fontsize=16, fontweight="bold"
        )
        ax.set_ylabel("Score", fontsize=14)
        ax.set_xlabel("Metric", fontsize=14)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            y_pos = min(height + 0.02, 1.05)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                rotation=0 if height > 0.9 else 0,
            )

        plt.tight_layout()
        return fig
