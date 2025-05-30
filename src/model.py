import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report


class KNN:
    def __init__(self, neighbors=None, test_size=0.2, random_state=42):
        """
        Αρχικοποίηση του μοντέλου KNN και των παραμέτρων του.

        Parameters:
            neighbors (int): Ο αριθμός των γειτόνων για το KNN. Αν δεν δοθεί, θα βρεθεί αυτόματα.
            test_size (float): Το ποσοστό των δεδομένων που θα χρησιμοποιηθούν για επικύρωση.
            random_state (int): Το seed για αναπαραγωγιμότητα.
        """
        self.response_column = "Ανταπόκριση"  # Ονομασία της στήλης που περιέχει την ανταπόκριση
        self.test_size = test_size  # Το ποσοστό των δεδομένων που θα χρησιμοποιηθούν για επικύρωση
        self.random_state = random_state  # Το seed
        self.best_n_neighbors = neighbors  # Ο αριθμός των γειτόνων για το KNN, αν έχει οριστεί
        self.results = []  # Αποθήκευση των μετρικών αποτελεσμάτων για κάθε fold
        self.detailed_results = [] # Αποθήκευση λεπτομερών αποτελεσμάτων (για καθε αριθμό γειτόνων) για κάθε fold
        self.X = None  # Τα χαρακτηριστικά των δεδομένων
        self.Y = None  # Η ανταπόκριση των δεδομένων
        self.X_train = None  # Τα χαρακτηριστικά των δεδομένων εκπαίδευσης
        self.y_train = None  # Η ανταπόκριση των δεδομένων εκπαίδευσης
        self.X_valid = None  # Τα χαρακτηριστικά των δεδομένων επικύρωσης
        self.y_valid = None  # Η ανταπόκριση των δεδομένων επικύρωσης
        self.preprocessor = None  # Ο προεπεξεργαστής των δεδομένων
        self.validation_metrics = None  # Οι μετρικές επικύρωσης του μοντέλου
        self.validation_metrics_str = ""  # Ένα string που περιέχει τις μετρικές επικύρωσης
        self.cv_validation_metrics = None  # Οι μετρικές επικύρωσης για cross-validation
        self.overall_validation_metrics = None  # Οι συνολικές μετρικές επικύρωσης
        self.final_model = None  # Το τελικό μοντέλο KNN μετά την εκπαίδευση

    def find_best_neighbors(self, on="accuracy", k_range=range(2, 21), fold_range=range(2, 11)):
        """
        Εύρεση του καλύτερου αριθμού γειτόνων για το KNN μέσω Grid Search με cross-validation σε διάφορα folds και εύρος αριθμού γειτόνων.

        Parameters:
            on (str): Η μετρική για την οποία θα γίνει η βελτιστοποίηση. Μπορεί να είναι "accuracy" ή "precision".
            k_range (range): Το εύρος των τιμών για τον αριθμό των γειτόνων που θα εξεταστούν.
            fold_range (range): Το εύρος των τιμών για τον αριθμό των folds στο cross-validation.
        """

        # Ορισμός του pipeline με τον preprocessor και τον classifier KNN
        knn = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", KNeighborsClassifier()),
            ]
        )

        # Ορισμός του grid search για τον αριθμό των γειτόνων
        param_grid = {
            "classifier__n_neighbors": k_range,
        }

        # Ορισμός των διαθέσιμων μετρικών χρησιμοποιηθούν για την αξιολόγηση του μοντέλου
        scoring = {
            "precision": "precision_macro",
            "accuracy": "accuracy",
        }

        if on not in scoring:
            raise ValueError(f"Invalid scoring method '{on}'. Available methods are: {', '.join(scoring.keys())}")

        # Εκτέλεση του grid search για κάθε fold στο εύρος που έχει οριστεί
        for c in fold_range:
            grid_search = GridSearchCV(
                knn,
                param_grid,
                cv=StratifiedKFold(n_splits=c),
                scoring=scoring,
                refit=on, # type: ignore
                n_jobs=-1,
                return_train_score=True,
            )

            # Εκπαίδευση του grid search με τα δεδομένα εκπαίδευσης
            grid_search.fit(self.X_train, self.y_train)

            # Αποθήκευση του καλύτερου αριθμού γειτόνων και των μετρικών
            current_best_n_neighbors = grid_search.best_params_[ "classifier__n_neighbors" ]

            # Εκπαίδευση του μοντέλου με τις καλύτερες παραμέτρους
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_valid)
            accuracy = accuracy_score(self.y_valid, y_pred)
            precision = precision_score(self.y_valid, y_pred, average="macro", zero_division=0) # type: ignore

            # Αποθήκευση των αποτελεσμάτων για το τρέχον fold
            self.results.append(
                {
                    "cv": c,
                    "neighbors": current_best_n_neighbors,
                    "cv_accuracy": accuracy,
                    "cv_precision": precision,
                }
            )

            # Αποθήκευση λεπτομερών αποτελεσμάτων (για καθε αριθμό γειτόνων) για το τρέχον fold
            for mean_precision, mean_accuracy, n in zip(
                grid_search.cv_results_["mean_test_precision"],
                grid_search.cv_results_["mean_test_accuracy"],
                grid_search.cv_results_["param_classifier__n_neighbors"],
            ):
                self.detailed_results.append(
                    {
                        "cv": c,
                        "neighbors": n,
                        "cv_precision": mean_precision,
                        "cv_accuracy": mean_accuracy,
                    }
                )

        # Μετατροπή των αποτελεσμάτων σε DataFrame και εύρεση του πιο συχνά εμφανιζόμενου αριθμού γειτόνων
        results_df = pd.DataFrame(self.results)
        self.best_n_neighbors = results_df["neighbors"].mode().iloc[0]

    def feed_data(self, train_data):
        """
        Προετοιμασία των δεδομένων για εκπαίδευση του μοντέλου KNN.

        Parameters:
            train_data (pd.DataFrame): Τα δεδομένα εκπαίδευσης που θα χρησιμοποιηθούν για την εκπαίδευση του μοντέλου.
        """

        # Διαχωρισμός των δεδομένων σε χαρακτηριστικά (X) και ανταπόκριση (y) (axis=1 για στήλες)
        self.X = train_data.drop(self.response_column, axis=1)
        self.y = train_data[self.response_column]

        # Διαχωρισμός των χαρακτηριστικών σε κατηγορικά και αριθμητικά
        categorical_cols = self.X.select_dtypes(include=["category"]).columns.tolist()
        numeric_cols = self.X.select_dtypes( include=["int64", "float64"]).columns.tolist()

        # Αρχικοποίηση του preprocessor με StandardScaler για αριθμητικά χαρακτηριστικά και OneHotEncoder για κατηγορικά χαρακτηριστικά
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numeric_cols,
                ),  # Κανονικοποίηση των αριθμητικών χαρακτηριστικών
                (
                    "cat",
                    OneHotEncoder(),
                    categorical_cols,
                ),  # Μετατροπή των κατηγορικών χαρακτηριστικών σε δυαδική μορφή
            ]
        )

        # Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και επικύρωσης
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state,
            stratify=self.y,
        )

    def fit(self):
        """
        Εκπαίδευση του μοντέλου KNN με τα δεδομένα εκπαίδευσης και τον καλύτερο αριθμό γειτόνων που έχει βρεθεί ή εχει οριστεί.

        Raises:
            ValueError: Αν δεν έχει οριστεί ο αριθμός γειτόνων ή αν δεν έχουν τροφοδοτηθεί τα δεδομένα εκπαίδευσης.
        """
        if self.best_n_neighbors is None:
            raise ValueError(
                "Ο αριθμός γειτόνων δεν έχει οριστεί. Καλέστε πρώτα τη μέθοδο find_best_neighbors() ή ορίστε τον αριθμό γειτόνων κατά την αρχικοποίηση του μοντέλου."
            )
        if self.X is None:
            raise ValueError(
                "Τα δεδομένα εκπαίδευσης δεν έχουν τροφοδοτηθεί στο μοντέλο. Καλέστε πρώτα τη μέθοδο feed_data()."
            )

        # Ορισμός του τελικού μοντέλου KNN
        self.final_model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=self.best_n_neighbors,
                    ),
                ),
            ]
        )

        # Εκπαίδευση του τελικού μοντέλου με τα δεδομένα εκπαίδευσης
        self.final_model.fit(self.X, self.y)

    def predict(self, new_data, output_path=None):
        """
        Κάνει προβλέψεις με το εκπαιδευμένο μοντέλο KNN για νέα δεδομένα.

        Parameters:
            new_data (pd.DataFrame): Τα νέα δεδομένα για τα οποία θα γίνουν προβλέψεις.
            output_path (str, optional): Το path για την αποθήκευση των αποτελεσμάτων σε αρχείο Excel. Αν δεν δοθεί, δεν θα αποθηκευτούν τα αποτελέσματα.

        Returns:
            pd.DataFrame: Ένα DataFrame που περιέχει τα νέα δεδομένα με τις αντίστοιχες προβλέψεις στην στήλη της ανταπόκρισης.

        Raises:
            ValueError: Αν το μοντέλο δεν έχει εκπαιδευτεί ή αν δεν έχουν τροφοδοτηθεί τα νέα δεδομένα.
        """
        if self.final_model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε πρώτα τη μέθοδο fit().")

        # Κανει την πρόβλεψη για τα νέα δεδομένα
        predictions_new = self.final_model.predict(new_data)

        # Αντιγραφή των νέων δεδομένων και προσθήκη των προβλέψεων στην αντίστοιχη στήλη
        result_df = new_data.copy()
        result_df[self.response_column] = predictions_new

        # Αν το output_path έχει δοθεί, αποθηκεύει τα αποτελέσματα σε αρχείο Excel
        if output_path:
            result_df.to_excel(output_path, index=False)

        # Επιστρέφει το DataFrame με τα αποτελέσματα
        return result_df

    def gen_metrics(self):
        """
        Δημιουργεί και αποθηκεύει τις μετρικές επικύρωσης του μοντέλου KNN.

        Raises:
            ValueError: Αν το μοντέλο δεν έχει εκπαιδευτεί.
        """
        if self.final_model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε πρώτα τη μέθοδο fit().")

        # Εκπαίδευση του τελικού μοντέλου με τα δεδομένα εκπαίδευσης
        self.final_model.fit(self.X_train, self.y_train)

        # Προβλέπει τις τιμές για το σύνολο επικύρωσης
        y_pred = self.final_model.predict(self.X_valid)
        # Υπολογίζει τις μετρικές επικύρωσης
        report = classification_report(self.y_valid, y_pred, output_dict=True)

        # Υπολογίζει τον πίνακα σύγχυσης οπου θα χρησιμοποιηθεί για τον υπολογισμό του class_specific_accuracy (δεν είναι διαθέσιμο στο classification_report)
        cm = confusion_matrix(self.y_valid, y_pred)

        # Υπολογίζει το accuracy για κάθε κλάση (yes, no)
        class_specific_accuracy = {}
        for i, label in enumerate(["yes", "no"]):
            TP = cm[i, i]
            total_class_instances = cm[i, :].sum()
            class_specific_accuracy[label] = (
                TP / total_class_instances if total_class_instances > 0 else 0
            )

        # Αποθηκεύει τις μετρικές επικύρωσης σε dict
        self.validation_metrics = {
            "Accuracy": report["accuracy"], # type: ignore
            "Precision": report["macro avg"]["precision"], # type: ignore
            "Yes Accuracy": class_specific_accuracy.get("yes", 0), # type: ignore
            "No Accuracy": class_specific_accuracy.get("no", 0), # type: ignore
            "Yes Precision": report["yes"]["precision"], # type: ignore
            "No Precision": report["no"]["precision"], # type: ignore
            "confusion_matrix": cm, # type: ignore
        }

        # Αποθηκεύσει των λεπτομερών μετρικών επικύρωσης σε dict
        self.cv_validation_metrics = {
            "best_neighbors_per_fold": pd.DataFrame(self.results),
            "all_neighbors_per_fold": pd.DataFrame(self.detailed_results),
        }

        # Αποθήκευση των συνολικών μετρικών επικύρωσης σε dict
        self.overall_validation_metrics = {
            "cv_validation_metrics": self.cv_validation_metrics,
            "validation_metrics": self.validation_metrics,
            "best_neighbors": self.best_n_neighbors,
        }

        # Δημιουργία του string με τις μετρικές επικύρωσης για αναφορά
        self.validation_metrics_str += "\nFinal Validation Metrics:\n"
        self.validation_metrics_str += (f"  • Validation Accuracy: {self.validation_metrics['Accuracy']:.4f}\n")
        self.validation_metrics_str += f"  • Validation Precision (macro): {self.validation_metrics['Precision']:.4f}\n"
        self.validation_metrics_str += "\n  • Class-specific Accuracy Scores:\n"
        self.validation_metrics_str += (f"    - Yes Accuracy: {self.validation_metrics['Yes Accuracy']:.4f}\n")
        self.validation_metrics_str += (f"    - No Accuracy: {self.validation_metrics['No Accuracy']:.4f}\n\n")
        self.validation_metrics_str += "\n  • Class-specific Precision Scores:\n"
        self.validation_metrics_str += f"     - Yes Precision (macro): {self.validation_metrics['Yes Precision']:.4f}\n"
        self.validation_metrics_str += f"     - No Precision (macro): {self.validation_metrics['No Precision']:.4f}\n"
