#set page(margin: 3em, height: auto)
#show link: it => underline(text(fill: blue)[#it])

#align(center)[#text(size: 20pt)[*Υλοποίηση Μοντέλου KNN*]]
#align(center)[#text(size: 12pt)[Ραφαήλ Ασλανίδης]]
#align(center)[#text(size: 8pt)[Τελικό Project ΠΛΗΠΡΟ -- Ελληνικό Ανοιχτό Πανεπιστήμιο -- 2025]]
\
= Εισαγωγή
Ο αλγόριθμος KNN (K-Nearest Neighbors) είναι ένας αλγόριθμος μηχανικής μάθησης που
χρησιμοποιείται τόσο για ταξινόμηση (classification) όσο και για παλινδρόμηση
(regression). Βασίζεται στην υπόθεση ότι δεδομένα με παρόμοια χαρακτηριστικά
βρίσκονται κοντά το ένα στο άλλο σε έναν πολυδιάστατο χώρο. Η απόδοση του KNN
εξαρτάται σε μεγάλο βαθμό από την επιλογή του κατάλληλου πλήθους γειτόνων K.

Στην παρούσα εργασία υλοποιείται ένας αλγόριθμος KNN με χρήση της
τη βιβλιοθήκη scikit-learn και χρησιμοποιεί την μέθοδο
#link("https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation:~:text=out%20cross%2Dvalidation.-,k%2Dfold%20cross%2Dvalidation,-%5Bedit%5D")[k-fold_cross-validation]
με διαφορετικούς συνδυασμούς τιμών του K και πλήθους fold. \ Στόχος είναι να
εντοπιστεί η τιμή του K που εμφανίζεται συχνότερα ως η βέλτιστη κατά τη
διάρκεια των επαναληπτικών πειραμάτων, ώστε να επιλεγεί ως η πλέον κατάλληλη
για το τελικό μοντέλο. Θα υπολογίζονται επίσης και οι μετρικές απόδοσης
του μοντέλου, όπως accuracy, precision, class-specific accuracy, και
class-specific precision.

Ο παραπάνω αλγόριθμος είναι το βασικό μέρος μιας GUI εφαρμογής που αναπτύσσεται στο πλαίσιο
του ομαδικού project του μαθήματος ΠΛΗΠΡΟ στο Ελληνικό Ανοιχτό Πανεπιστήμιο για την
πρόβλεψη της ανταπόκρισης πελατών σε καμπάνιες marketing με χρήση μηχανικής μάθησης.

= Μεθοδολογία και Συνεισφορά
Ο αλγόριθμος KNN βασίζεται στον υπολογισμό αποστάσεων
μεταξύ σημείων δεδομένων. Συνήθως (όπως και σε αυτήν την εργασία)
χρησιμοποιείται η απόσταση
#link("https://en.wikipedia.org/wiki/Euclidean_distance")[Euclidean] όπου
είναι και το default μέτρο απόστασης στη βιβλιοθήκη scikit-learn.

Η σωστή επιλογή του παραμέτρου K είναι κρίσιμη για την απόδοση του μοντέλου. Μικρές
τιμές K οδηγούν overfitting, ενώ πολύ μεγάλες τιμές
μπορεί να οδηγήσουν σε underfitting, το optimal είναι κάπου ενδιάμεσα.

Το #link("https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation:~:text=out%20cross%2Dvalidation.-,k%2Dfold%20cross%2Dvalidation,-%5Bedit%5D")[k-fold
cross-validation], είναι μια μέθοδος που επιτρέπει την αποτίμηση της απόδοσης
ενός μοντέλου με μεγαλύτερη σταθερότητα. Το dataset χωρίζεται σε
k ίσα μέρη (folds), και το μοντέλο εκπαιδεύεται και αξιολογείται k φορές, κάθε
φορά με διαφορετικό fold ως σύνολο δοκιμής.

Η διαδικασία αυτή εφαρμόζεται για κάθε τιμή του K (γειτόνων), και η απόδοση του μοντέλου
καταγράφεται ξεχωριστά για κάθε fold. Σκοπός είναι να αναγνωριστεί η τιμή του K που
εμφανίζεται συχνότερα ως η βέλτιστη.

=== Μέθοδοι της Κλάσης KNN
Συνοπτικά, υλοποίησα τον ακόλουθο αλγόριθμο: \
- ```py find_best_neighbors()```
  Για κάθε διαφορετικό πλήθος fold (π.χ. 5, 10) τρέχουμε τον αλγόριθμο KNN με διαφορετικές τιμές του K και
  καταγράφουμε την απόδοση του μοντέλου. Στο τέλος, αναλύουμε τα αποτελέσματα
  και επιλέγουμε την τιμή του K που εμφανίζεται συχνότερα ως η βέλτιστη με τη
  βοήθεια της μεθόδου mode() της βιβλιοθήκης pandas.
Πέρα από την υλοποίηση του παραπάνω αλγορίθμου, υλοποίησα και τις ακόλουθες μεθόδους:
- ```py __init__()```
  Αρχικοποιεί τα attributes της κλάσης KNN.
- ```py feed_data()```
  Φορτώνει τα δεδομένα εκπαίδευσης και
  τα μετατρέπει σε μορφή κατάλληλη (numerical & categorical) για να χρησιμοποιηθούν
  από τον preprocessor του KNN.
- ```py fit()```
  Εκπαιδεύει το μοντέλο KNN με τα δεδομένα εκπαίδευσης.
- ```py predict()```
  Κάνει πρόβλεψη για νέα δεδομένα
  και αποθηκεύει τα αποτελέσματα σε αρχείο εάν δοθεί το `output_path`.
- ```py gen_metrics()```
  Υπολογίζει και αρχικοποιεί τις μετρικές απόδοσης του μοντέλου,
  όπως accuracy, precision, class-specific accuracy, και class-specific precision σε dicts
  αλλά και σε ένα string για εύκολη εμφάνιση.

Επιπλέον, δημιούργησα και μια κλάση `Plotter` για την οπτικοποίηση της απόδοσης του μοντέλου
ανά συνδυασμό τιμών K και folds σε περίπτωση που ο χρήστης επιλέξει να χρησιμοποιήσει την
μέθοδο `find_best_neighbors()`.

=== Μέθοδοι της Κλάσης Plotter
Η παρακάτω κλάση δημιουργήθηκε για να διευκολύνει την οπτικοποίηση των αποτελεσμάτων
στο ομαδικό paper και περιλαμβάνει τις ακολούθες μεθόδους:
- ```py __init__()```
  Αρχικοποιεί τα attributes της κλάσης Plotter.
- ```py plot_neighbors_vs_metric_per_fold()```
  Δημιουργεί ένα graph που απεικονίζει την απόδοση του μοντέλου ανά
  συνδυασμό τιμών K και folds για μια συγκεκριμένη μετρική (π.χ. accuracy).
- ```py plot_trisurf_neighbor_vs_metric_per_fold()```
  Δημιουργεί ένα 3D graph που απεικονίζει την απόδοση του μοντέλου
  ανά συνδυασμό τιμών K και folds για μια συγκεκριμένη μετρική (π.χ. accuracy).
- ```py plot_mean_metric_per_fold()```
  Δημιουργεί ένα graph που απεικονίζει την μέση απόδοση του μοντέλου ανά fold
  για μια συγκεκριμένη μετρική (π.χ. accuracy).

Επίσης, βοήθησα στη δημιουργία keybind για την έξοδο του GUI της εφαρμογής στον
κώδικα του Κώνσταντινου Πιτσαρή, καθώς και με κάποια προβλήματα που προέκυψαν κατά το
setup του project στον υπολογιστή της Αντωνίας Κρανίτσας. Ακόμη, ως συντονιστής της ομάδας, βοήθησα στην οργάνωση των εργασιών και στην δημιουργία
ενός GitHub repository για την αποθήκευση του κώδικα και των αρχείων της εργασίας.

= Ώρες Εργασίας
- Εισαγωγική μελέτη του αλγορίθμου KNN και των τεχνικών cross-validation και grid search. ($approx$ *1* ώρα)
- Υλοποίηση της κλάσης `KNN` με χρήση της βιβλιοθήκης scikit-learn. ($approx$ *6* ώρες)
- Δημιουργία της κλάσης `Plotter` για την οπτικοποίηση των αποτελεσμάτων. ($approx$ *2* ώρες)
- Τεκμηρίωση του κώδικα και των μεθόδων με σχόλια και docstrings. ($approx$ *1* ώρα)
- Συγγραφή του ατομικού paper. ($approx$ *2* ώρες)
- Συνεισφορά στη συγγραφή του ομαδικού paper. ($approx$ *5* ώρες)

= Βιβλιογραφία
- Scikit-learn
  - #link("https://scikit-learn.org/stable/api/index.html")[API]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html")[Pipeline]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html")[ColumnTransformer]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html")[KNeighborsClassifier]
  - #link("https://scikit-learn.org/stable/modules/preprocessing.html")[Preprocessing]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html")[OneHotEncoder]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html")[StandardScaler]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html")[TrainTestSplit]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html")[GridSearchCV]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html")[StratifiedKFold]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html")[AccuracyScore]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html")[PrecisionScore]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html")[ConfusionMatrix]
  - #link("https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html")[ClassificationReport]

- *Pandas*
  - #link("https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html")[DataFrame]

- *Matplotlib*
  - #link("https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html")[Pyplot]

- *Seaborn*
  - #link("https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn-lineplot")[Lineplot]
  - #link("https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot")[Barplot]

- *LLMs*
  - Χρησιμοποιήθηκε το ChatGPT της OpenAI ως βοηθητικό εργαλείο μάθησης για την κατανόηση εννοιών
    που δεν μου ήταν ξεκάθαρες κατά την μελέτη του documentation των βιβλιοθηκών. Δεν υπήρξε αντιγραφή κώδικα.

- *Βίντεο*
  - #link("https://www.youtube.com/watch?v=0p0o5cmgLdE")[[YouTube] K Nearest Neighbors | Intuitive explained | Machine Learning Basics]
  - #link("https://www.youtube.com/watch?v=HVXime0nQeI")[[YouTube] StatQuest: K-nearest neighbors, Clearly Explained]
  - #link("https://www.youtube.com/watch?v=fSytzGwwBVw")[[YouTube] Machine Learning Fundamentals: Cross Validation]
  - #link("https://www.youtube.com/watch?v=G-fXV-o9QV8")[[YouTube] Hyperparameters Tuning: Grid Search vs Random Search]

- *Extra*
  - #link("https://typst.app/docs/")[Typst Documentation]

- *Bonus*
  - #link("https://www.youtube.com/watch?v=ozhHrzOXZkI")[Terry A. Davis]
