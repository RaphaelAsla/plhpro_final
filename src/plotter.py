import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self, metrics):
        """
        Αρχικοποιεί την κλάση Plotter με τα δεδομένα των μετρικών.

        :Parameters 
            metrics (dict): Ένα dict που περιέχει τις μετρικές από την εκπαίδευση του μοντέλου.
        """
        self.cv_metrics = (
            metrics["cv_validation_metrics"]["best_neighbors_per_fold"],
            metrics["cv_validation_metrics"]["all_neighbors_per_fold"]
        )
        self.best_neighbors = metrics["best_neighbors"]
        

    def plot_neighbors_vs_metric_per_fold(self, metric="accuracy", output_path=None):
        """
        Δημιουργεί ένα graph που απεικονίζει τη σχέση μεταξύ του αριθμού των γειτόνων και της μετρικής για κάθε fold.

        :Parameters 
            metric (string) : Η μετρική που θα απεικονιστεί (π.χ. "accuracy", "precision").
            output_path (string): Το path για αποθήκευση του graph. Αν είναι None, το γράφημα θα εμφανιστεί στην οθόνη.
        """
        metric = "cv_" + metric
        title = metric[3].upper() + metric[3:]

        df = self.cv_metrics[1]
        df["cv"] = df["cv"].astype(str) # παραλείπονται κάποια folds χωρίς αυτό, δεν είμαι σίγουρος γιατί

        sns.lineplot(data=df, x="neighbors", y=metric, hue="cv", marker="o")

        plt.axvline(self.best_neighbors, color="red", linestyle="--", linewidth=1)

        plt.xlabel("Number of Neighbors")
        plt.ylabel(f"{title}")
        plt.title(f"{title} vs Number of Neighbors per Fold")
        plt.grid(True)
        plt.legend(title="Folds", loc='upper right', fontsize='small', labelspacing=0.3) 
        plt.savefig(output_path) if output_path else plt.show()
        plt.clf()

    def plot_mean_metric_per_fold(self, metric="accuracy", output_path=None):
        """
        Δημιουργεί ένα graph που απεικονίζει τη μέση τιμή της μετρικής για κάθε fold.
        :Parameters 
            metric (string): Η μετρική που θα απεικονιστεί (π.χ. "accuracy", "precision").
            output_path (string): Το path για αποθήκευση του graph. Αν είναι None, το γράφημα θα εμφανιστεί στην οθόνη.
        """
        metric = "cv_" + metric
        title = metric[3].upper() + metric[3:]

        df = self.cv_metrics[1]
        df["cv"] = df["cv"].astype(str) # skipping some folds otherwise

        sns.barplot(data=df, x="cv", y=metric, hue="cv")

        plt.xlabel("Number of Folds")
        plt.ylabel(f"{title}")
        plt.title(f"{title} vs Mean per Fold")
        plt.grid(True)

        plt.ylim(df[metric].min() - 0.005, df[metric].max() + 0.005)

        plt.savefig(output_path) if output_path else plt.show()
        plt.clf()

    def plot_trisurf_metric_per_fold(self, metric="accuracy", output_path=None):
        """
        Δημιουργεί ένα 3D graph που απεικονίζει τη σχέση μεταξύ του αριθμού των γειτόνων, της μετρικής και της ακρίβειας για κάθε fold.
        :Parameters 
            metric (string): Η μετρική που θα απεικονιστεί (π.χ. "accuracy", "precision").
            output_path (string): Το path για αποθήκευση του graph. Αν είναι None, το γράφημα θα εμφανιστεί στην οθόνη.
        """
        metric = "cv_" + metric
        title = metric[3].upper() + metric[3:]

        df = self.cv_metrics[1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  

        ax.scatter(df['neighbors'], df['cv'], df['cv_accuracy'], cmap="viridis", edgecolor="none") # type: ignore

        ax.set_xlabel('neighbors')
        ax.set_ylabel('cv')
        ax.set_zlabel('accuracy') # type: ignore

        plt.title(f"{title} vs Number of Neighbors per Fold 3D")

        plt.savefig(output_path) if output_path else plt.show()

