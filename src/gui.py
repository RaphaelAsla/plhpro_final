"""
CampaignPredictionApp: Γραφικό Περιβάλλον Διεπαφής Χρήστη για φόρτωση ιστορικών
δεδομένων, εκπαίδευση μοντέλου KNN, φόρτωση νέων δεδομένων, πρόβλεψη και
αποθήκευση αποτελεσμάτων.

Εκκίνηση: app = CampaignPredictionApp(width, height); app.run()
width και height δίνονται σε pixels.

Απαιτείται εγκατάσταση των εξής πακέτων:
    tkinter
    sv_ttk
    pandas
"""
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext, ttk
import sv_ttk
import pandas as pd
from model import KNN

class CampaignPredictionApp:
    """ 
    Κλάση υλοποίησης Γραφικού Περιβάλλοντος Εφαρμογής Προβλέψεων Ανταπόκρισης
    Πελατών με χρήση μοντέλου μηχανικής εκμάθησης K-nn.    
    Μεταβάσεις κατάστασης εφαρμογής:
    1. Αρχή -> load_past -> training_data_loaded
    2. training_data_loaded -> train -> model_trained
    3. model_trained -> load_new -> predictions_data_loaded
    4. predictions_data_loaded -> predict -> predictions_made
    5. predictions_made -> save -> restart (όλα τα flags False και όλα τα 
                                    dataframes και instances αρχικοποιούνται)
    """
    def __init__(self, width, height):
        # Αρχικοποίηση της εφαρμογής GUI
        self.master = tk.Tk() # Δημιουργία "root" μέσα στη κλάση

        sv_ttk.set_theme('dark') #Εφαρμογή θέματος sv_ttk

        self.master.title("Εφαρμογή Προβλέψεων Ανταπόκρισης Νέας Καμπάνιας") # Τίτλος Παραθύρου Εφαρμογής
        self.master.geometry(f"{width}x{height}") # Διαστάσεις Παραθύρου Εφαρμογής - Παίρνει όρισμα στη main
        self.master.minsize(800,600) # Ελάχιστες διαστάσεις του κεντρικού παραθύρου

        self.past_campaign_data = None # το df με τα στοιχεία της προηγούμενης καμπάνιας
        self.new_campaign_data = None # το df με τα στοιχεία της νέας καμπάνιας
        self.knn_model = None # το instance της Κ-ΝΝ που είσαγεται
        self.predictions_df = None # το df με τα στοιχεία της πρόβλεψης

        #Αρχικοποίηση κουμπιών
        self.btn_load_past = None
        self.btn_load_new = None
        self.btn_train = None
        self.btn_predict = None
        self.btn_save = None

        # Flags για διαχείριση κατάστασης κουμπιών
        # Κάθε flag σηματοδοτεί ποιό στάδιο έχει ολοκληρωθεί
        self.training_data_loaded = False
        self.model_trained = False
        self.predictions_data_loaded = False
        self.predictions_made = False

        # Δημιουργία των widgets της εφαρμογής
        self._create_widgets()
        self._log("Ξεκινήστε πρώτα με τη Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας.\n")
        # Ενημέρωση Κατάστασης Κουμπιών
        self._update_button_states()

    def _create_widgets(self):
        """
        Δημιουργία των στοιχείων του Γραφικού Περιβάλλοντος.
        Χρησιμοποιούμε pack() για να ορίσουμε το γενικό πλαίσιο,
        και grid() για ακριβή διάταξη κουμπιών μέσα σε αυτό.
        """
        button_frame = ttk.Frame(self.master)    # Πλαίσιο για τα κουμπιά
        button_frame.pack(
            padx=10, # Προσθέτει οριζόντιο χώρο εκτός του πλαισίου του κουμπιού (δεξία & αριστερά)
            pady=10, # Προσθέτει κατακόρυφο χώρο εκτός του πλαισίου του κουμπιού (πάνω & κάτω)
            fill=tk.X, # "Γεμίζει" όλο τον οριζόντιο χώρο για να καταλαμβάνει ολό το ελεύθερο πλάτος
            expand=False # Περιορίζει το πλαίσιο των κουμπιών για να μην "κλέβουν" χώρο απο την text_area
            )
        button_frame.grid_columnconfigure(0, weight=1)  # ρυθμίζει το πως επεκτείνονται οι στήλες των πλαισίων
        button_frame.grid_columnconfigure(1, weight=1)

        # 1. Κουμπί Φόρτωσης Παλαιών Δεδομένων Καμπάνιας
        self.btn_load_past = ttk.Button(
            button_frame,
            text="1. Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας", # Τίτλος κουμπιού
            command=self.load_past_campaign_data # Εντολή που εκτελείται όταν πατιέται το κουμπί
            )
        self.btn_load_past.grid(
            row=0,
            column=0,
            columnspan=2, # Χώρος που καταλαμβάνει η στήλη
            padx=5, # Προσθέτει οριζόντιο εσωτερικό περιθώριο (αριστερά/δεξιά) γύρω από το κουμπί
            pady=5, # Προσθέτει κάθετο εσωτερικό περιθώριο (πάνω/κάτω) γύρω από το κουμπί
            sticky="nsew" # Κάνει τα κουμπιά να "κολλάνε" στις διαστάσεις του παραθύρου
            )

        # 2. Κουμπί Εκπαίδευσης Μοντέλου
        self.btn_train = ttk.Button(
            button_frame,
            text="2. Εκπαίδευση Μοντέλου Πρόβλεψης",
            command=self.on_train
            )
        self.btn_train.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )

        # 3. Κουμπί Φόρτωσης Νέων Δεδομένων για Πρόβλεψη
        self.btn_load_new = ttk.Button(
            button_frame,
            text="3. Φόρτωση Δεδομένων Νέας Καμπάνιας",
            command=self.load_new_campaign_data
            )
        self.btn_load_new.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )

        # 4. Κουμπί Πρόβλεψης Ανταπόκρισης Νέων Πελατών
        self.btn_predict = ttk.Button(
            button_frame,
            text="4. Πρόβλεψη Ανταπόκρισης Νέων Πελατών",
            command=self.on_predict
            )
        self.btn_predict.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )

        # 5. Κουμπί Αποθήκευσης Πρόβλεψης
        self.btn_save = ttk.Button(
            button_frame,
            text="5. Αποθήκευση Πρόβλεψης",
            command=self.save_predictions_wrapper
            )
        self.btn_save.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )

        # Περιοχή κειμένου για εμφάνιση αποτελεσμάτων
        self.text_area = scrolledtext.ScrolledText(
            self.master,
            width=100,
            height=20,
            font=("Trebuchet MS",12),
            )
        self.text_area.pack(
            padx=15,
            pady=15,
            fill=tk.BOTH,
            expand=True
            )

        # Απενεργοποίηση δυνατότητας εγγραφής κειμένου
        self.text_area.config(state=tk.DISABLED)

    def _update_button_states(self):
        """
        Ιδιωτική μέθοδος που ενεργοποιεί και απενεργοποίει τη κατάσταση των
        κουμπιών της εφαρμογής, ανάλογα με το στάδιο στο οποίο βρίσκεται
        η διαδικασία.
        
        Κάθε ένα if-block ελέγχει την κατάσταση στην οποία βρίσκεται η διαδικασία
        της εφαρμογής και ενεργοποιεί / απενεργοποίει τα κουμπία αναλόγως.
        
        Κατάσταση Flag            | Κατάσταση Κουμπιών
        1. None                     -> Μόνο το κουμπί Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας ΕΝΕΡΓΟ
        2. training_data_loaded     -> Μόνο το κουμπί Εκπαίδευση Μοντέλου Πρόβλεψης ΕΝΕΡΓΟ
        3. model_trained            -> Μόνο το κουμπί Φόρτωση Δεδομένων Νέας Καμπάνιας ΕΝΕΡΓΟ
        4. predictions_data_loaded  -> Μόνο το κουμπί Πρόβλεψης Ανταπόκρισης Νέων Πελατών ΕΝΕΡΓΟ
        5. predictions_made         -> Μόνο το κουμπί Αποθήκευσης Πρόβλεψης ΕΝΕΡΓΟ
        """
        active, inactive = 'normal', 'disabled'
        # Στάδιο 1 - Φόρτωση Ιστορικών Δεδομένων
        if not self.training_data_loaded and not self.model_trained and not self.predictions_data_loaded and not self.predictions_made:
            self.btn_load_past.config(state=active)
            self.btn_train.config(state=inactive)
            self.btn_load_new.config(state=inactive)
            self.btn_predict.config(state=inactive)
            self.btn_save.config(state=inactive)
            return
        # Στάδιο 2 - Εκπαίδευση Μοντέλου Πρόβλεψης
        if self.training_data_loaded and not self.model_trained:
            self.btn_load_past.config(state=inactive)
            self.btn_train.config(state=active)
            self.btn_load_new.config(state=inactive)
            self.btn_predict.config(state=inactive)
            self.btn_save.config(state=inactive)
            return
        # Στάδιο 3 - Φόρτωση Δεδομένων για την πρόβλεψη
        if self.model_trained and not self.predictions_data_loaded:
            self.btn_load_past.config(state=inactive)
            self.btn_train.config(state=inactive)
            self.btn_load_new.config(state=active)
            self.btn_predict.config(state=inactive)
            self.btn_save.config(state=inactive)
            return
        # Στάδιο 4 - Πρόβλεψη Ανταπόκρισης
        if self.predictions_data_loaded and not self.predictions_made:
            self.btn_load_past.config(state=inactive)
            self.btn_train.config(state=inactive)
            self.btn_load_new.config(state=inactive)
            self.btn_predict.config(state=active)
            self.btn_save.config(state=inactive)
            return
        # Στάδιο 5 - Αποθήκευση αποτελεσμάτων
        if self.predictions_made:
            self.btn_load_past.config(state=inactive)
            self.btn_train.config(state=inactive)
            self.btn_load_new.config(state=inactive)
            self.btn_predict.config(state=inactive)
            self.btn_save.config(state=active)
            return

    def _log(self, message):
        """Ιδιωτική μέθοδος για εγγραφή κειμένου στην περιοχή κειμένου"""        
        self.text_area.config(state=tk.NORMAL) # Ενεργοποίηση δυνατότητας εγγραφής κειμένου
        self.text_area.insert(tk.END, message + "\n") # Εμφανίζει το κείμενο message
        self.text_area.see(tk.END) # Μετακίνηση στο τέλος του κειμένου
        self.text_area.config(state=tk.DISABLED) # Ξανά απενεργοποίηση δυνατότητας εγγραφής

    def _load_data(self,title):
        """
        Εσωτερική μέθοδος κλάσης φόρτωσης δεδομένων απο αρχείο Excel +
        Διαχείριση Εξαιρέσεων.
        H μορφή που αναμένεται να έχει το Excel αρχείο που φορτώνεται είναι η
        εξής:
        [Ηλικία | Φύλο | Περιοχή |	Email | Χρήση Κινητού |
        Logins τις τελευταίες 4 εβδομάδες | Logins τους τελευταίους 6 μήνες |
        Αγορές τις τελευταίες 4 εβδομάδες | Αγορές τους τελευταίους 6 μήνες |
        Σύνολο Αγορών | Ανταπόκριση].
        Κάθε στήλη είναι ΥΠΟΧΡΕΩΤΙΚΗ για να τρέξει ορθά η εφαρμογή.
        Αλλάζει το index σε μορφή 'Πελάτης Ν', έαν έχει το default RangeIndex.
        """
        file_path= filedialog.askopenfilename(
            title=title,
            filetypes=[("Excel files", "*.xlsx")]
            )
        if not file_path:
            messagebox.showwarning("Προσοχή!", "Δεν επιλέχθηκε αρχείο.")
            return None
        try:
            df = pd.read_excel(file_path)
            # Έλεγχος εάν το DataFrame είναι κενό
            if df.empty:
                messagebox.showerror("Σφάλμα!",f"Το αρχείο είναι άδειο ή είναι μη έγκυρου τύπου: {file_path}")
                return None
            new_index = {i: f"Πελάτης {i+1}" for i in range(len(df))} # λεξικό δυναμικό ανάλογα με το μέγεθος του df
            if isinstance(df.index, pd.RangeIndex): # εάν το df έχει το default index, το μεταονομάζει
                df = df.rename(index=new_index)
            # Επιστρέφουμε το DataFrame με νέο index 'Πελάτης N'
            return df
        except FileNotFoundError:
            messagebox.showerror("Σφάλμα!", f"Το αρχείο δεν βρέθηκε: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            messagebox.showerror("Σφάλμα!", f"Το αρχείο είναι εντελώς άδειο: {file_path}")
            return None
        except pd.errors.ParserError:
            messagebox.showerror("Σφάλμα!", f"Σφάλμα τιμής κατά την ανάγνωση του αρχείου Excel: {file_path}")
            return None
        except ValueError as ve:
            messagebox.showerror("Σφάλμα!", f"Σφάλμα τιμής κατά την ανάγνωση του Excel: {file_path}\n{str(ve)}")
            return None
        except Exception as e:
            messagebox.showerror("Σφάλμα!", f"Προέκυψε άγνωστο σφάλμα κατά τη φόρτωση του αρχείου Excel: {file_path}\n{str(e)}")
            return None

    def load_past_campaign_data(self):
        """
        Μέθοδος Φόρτωσης Δεδομένων Προηγούμενης Καμπάνιας
        """
        self._log("\n=== Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας ===\n")
        temp_data = self._load_data("Επιλέξτε αρχείο δεδομένων προηγούμενης καμπάνιας")
        if temp_data is not None:
            # Ενημέρωση flags
            self.training_data_loaded = True
            self.model_trained = False
            self.predictions_data_loaded = False
            self.predictions_made = False
            # Αρχικοποίηση μεταβλητών σε περίπτωση νέων δεδομένων
            self.knn_model = None
            self.predictions_df = None

            # Ενημέρωση κατάστασης κουμπιών
            self._update_button_states()

            self.past_campaign_data = temp_data
            messagebox.showinfo("Επιτυχία!", "Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς.")
            messagebox.showinfo("Επόμενο Βήμα","Προχωρήστε στην Εκπαίδευση Μοντέλου Πρόβλεψης.")
            messagebox.showwarning("Προειδοποίηση!", "Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα.")
            self._log("\nΕπόμενο βήμα: Προχωρήστε στην εκπαίδευση του μοντέλου πρόβλεψης.")
            self._log("Προειδοποίηση: Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα.")
            self._log("Το γραφικό περιβάλλον της εφαρμογής πιθανώς να 'παγώσει' κατά τη διάρκεια της εκπαίδευσης.")
        else:
            self._log("Η φόρτωση των παλαιών δεδομένων απέτυχε ή ακυρώθηκε.")

    def load_new_campaign_data(self):
        """
        Μέθοδος Φόρτωσης Δεδομένων της Νέας Καμπάνιας
        """
        self._log("\n=== Φόρτωση Δεδομένων Νέας Καμπάνιας ===\n")
        temp_data = self._load_data("Επιλέξτε αρχείο δεδομένων νέας καμπάνιας")
        if temp_data is not None:
            # Ενημέρωση flags
            self.predictions_data_loaded = True
            self.predictions_made = False
            # Αρχικοποίηση df σε περίπτωση νέων δεδομένων
            self.predictions_df = None
            # Ενημέρωση κατάστασης κουμπιών
            self._update_button_states()

            self.new_campaign_data = temp_data
            messagebox.showinfo("Επιτυχία!", "Τα δεδομένα της νέας καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("Τα δεδομένα της νέας καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("\n=================================================\n")
            messagebox.showinfo("Επόμενο Βήμα", "Προχωρήστε στην Πρόβλεψη Ανταπόκρισης Νέων Πελατών.")
            self._log("Επόμενο βήμα: Προχωρήστε στην πρόβλεψη ανταπόκρισης νέων πελατών.")
        else:
            self._log("Η φόρτωση των νέων δεδομένων απέτυχε ή ακυρώθηκε.")
            self._log("\n=================================================\n")

    def _save_predictions(self, df_to_save):
        """ 
        Εσωτερική Μέθοδος για την Αποθήκευση των Προβλέψεων σε νέο αρχείο Excel
        Παίρνει το dataframe προς αποθήκευση ως όρισμα.
        """
        if df_to_save is None or df_to_save.empty:
            messagebox.showerror("Σφάλμα!", "Δεν υπάρχουν δεδομένα πρόβλεψης προς αποθήκευση.")
            return False
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Αποθήκευση Προβλέψεων"
            )
        if not save_path:
            messagebox.showwarning("Ακύρωση", "Η αποθήκευση ακυρώθηκε.")
            self._log("Η αποθήκευση ακυρώθηκε.")
            return False
        try:
            df_to_save.to_excel(
                save_path,
                index=True # Κρατάμε τα index στα οποία αλλάξαμε το όνομα προηγουμένως
                )
            messagebox.showinfo("Επιτυχία", f"Οι προβλέψεις αποθηκεύτηκαν με επιτυχία στο:\n{save_path}")
            return True
        except Exception as e:
            messagebox.showerror("Σφάλμα Αποθήκευσης!", f"Δεν ήταν δυνατή η αποθήκευση των προβλέψεων.\nΣφάλμα:{str(e)}")
            self._log(f"Σφάλμα κατά την αποθήκευση στο {save_path}: {str(e)}")
            return False

    def save_predictions_wrapper(self):
        """
        Βοηθητική Μέθοδος που καλείται απο το Κουμπί Αποθήκευσης.
        Εάν υπάρχουν προβλέψεις, επιχειρούμε την αποθήκευση - αλλιώς ειδοποιούμε τον χρήστη καταλλήλως.
        Επιστρέφει success μόνο εάν ο χρήστης ολοκληρώσει απροβλημάτιστα
        την επιλογή path, χωρίς εμφάνιση εξαιρέσεων.        
        """
        self._log("\n=== Aποθήκευση Προβλέψεων ===")
        if self.predictions_df is not None:
            success = self._save_predictions(self.predictions_df)
            if success:
                # Επαναφορά όλων σε αρχική κατάσταση
                self.past_campaign_data = None
                self.new_campaign_data = None
                self.knn_model = None
                self.predictions_df = None

                # Επαναφορά των flags
                self.training_data_loaded = False
                self.model_trained = False
                self.predictions_data_loaded = False
                self.predictions_made = False

                # Ενημέρωση κουμπιών και log
                self._update_button_states()
                messagebox.showinfo('Επαναφορά', 'Μπορείτε να ξεκινήσετε ξανά από την αρχή.')
            self._log("\n===============ΤΕΛΟΣ ΠΡΟΓΡΑΜΜΑΤΟΣ===============")
            self._log("\n=== Το πρόγραμμα επανήλθε στην αρχική κατάσταση. ===")
        else:
            messagebox.showwarning("Προσοχή!", "Δεν έχουν δημιουργηθεί προβλέψεις προς αποθήκευση.")
            self._log("Αποτυχία αποθήκευσης: Δεν υπάρχουν διαθέσιμες προβλέψεις.")

    def on_train(self):
        """
        Μέθοδος που εκτελείται όταν πατηθεί το κουμπί Εκπαίδευσης.
        Επαλήθευση Δεδομένων Εκπαίδευσης -> Τροφοδοσία Μοντέλου -> Διαχείριση 
        Εξαιρέσεων.
        """
        self._log("\n=== Έναρξη Διαδικασίας Εκπαίδευσης Μοντέλου Πρόβλεψης Κ-nn ===\n")
        if self.past_campaign_data is None: # Έλεγχος εάν έχουν φορτωθεί δεδομένα εκπαίδευσης
            messagebox.showerror("Σφάλμα!", "Δεν έχουν φορτωθεί δεδομένα εκπαίδευσης.")
            self._log("Σφάλμα: Απαιτούνται δεδομένα εκπαίδευσης.")
            return
        try:
            # 1. Αρχικοποίηση του K-NN μοντέλου
            # 2. Τροφοδοσία δεδομένων εκπαίδευσης
            # 3. Βελτιστοποίηση παραμέτρου k με cross-validation
            # 4. Εκπαίδευση τελικού μοντέλου με το βέλτιστο k
            self._log("Aρχικοποίηση επεξεργαστή K-nn...")
            self.knn_model = KNN(neighbors=None, test_size=0.2, random_state=42)
            self._log("Τροφοδότηση δεδομένων εκπαίδευσης στο μοντέλο...")
            self.knn_model.feed_data(self.past_campaign_data)
            self._log("Εύρεση βέλτιστου αριθμού γειτόνων (k)...")
            k_range = range(2, 16) # Προσαρμόζουμε το εύρος ανάλογα (μεγαλύτερο εύρος γειτόνων=αργότερη εκτέλεση)
            fold_range = range(2, 8) # Προσαρμόζουμε το εύρος ανάλογα (μεγαλύτερο εύρος folds=αργότερη εκτέλεση)
            self.knn_model.find_best_neighbors(k_range=k_range, fold_range=fold_range)
            self.model_trained = True
            self._update_button_states()
            self._log("Ολοκληρώθηκε η διαδικασία εύρεσης βέλτιστου αριθμού γειτόνων (k)")
            self._log(f"\n    -Βέλτιστος αριθμός γειτόνων (k):{self.knn_model.best_n_neighbors}\n")
            self._log("Εκπαίδευση τελικού μοντέλου με τα πλήρη δεδομένα εκπαίδευσης...")
            self.knn_model.fit()
            self._log("\n   -Το τελικό μοντέλο εκπαιδεύτηκε με k = " + str(self.knn_model.best_n_neighbors))
            messagebox.showinfo("Εκπαίδευση Ολοκληρώθηκε!", f"Βέλτιστο k = {self.knn_model.best_n_neighbors}\n")
            self._log("\n=================================================\n")
            self.predictions_df = None # Ακύρωση τυχόν προηγούμενων προβλέψεων τώρα που το μοντέλο επανεκπαιδεύτηκε
            messagebox.showinfo("Επόμενο Βήμα", "Προχωρήστε στη φόρτωση των δεδομένων νέας καμπάνιας.")
            self._log("Επόμενο βήμα: Προχωρήστε στη φόρτωση των δεδομένων νέας καμπάνιας.")
        except ValueError as ve: # Ανεπαρκή ή λάθος μορφή δεδομένων
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Προέκυψε σφάλμα τιμής κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(ve)}")
            self._log(f"Σφάλμα (ValueError) κατά την εκπαίδευση: {str(ve)}")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
            self.model_trained = False
            self.training_data_loaded = True
        except Exception as e:
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Προέκυψε άγνωστο σφάλμα κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(e)}")
            self._log(f"Άγνωστο σφάλμα κατά την εκπαίδευση: {str(e)}\n")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
            self.model_trained = False
            self.training_data_loaded = True

    def on_predict(self):
        """
        Μέθοδος που εκτελείται όταν πατιέται το Κουμπί Πρόβλεψης.
        Επαλήθευση Ύπαρξης Μοντέλου Πρόβλεψης -> Πρόβλεψη Ανταπόκρισης ->
        Διαχείριση Εξαιρέσεων.
        """
        self._log("\n=== Έναρξη Διαδικασίας Πρόβλεψης ===\n")
        if self.knn_model is None: # Έλεγχος εάν υπάρχει διαθέσιμο εκπαιδευμένο μοντέλο πρόβλεψης
            messagebox.showerror("Σφάλμα!", "Δεν υπάρχει εκπαιδευμένο μοντέλο πρόβλεψης.\nΕκτελέστε πρώτα την εκπαίδευση.")
            self._log("Σφάλμα: Απαιτείται εκπαιδευμένο μοντέλο για την πρόβλεψη.")
            return

        if self.new_campaign_data is None:  # Έλεγχος εάν έχουν φορτωθεί νέα δεδομένα για πρόβλεψη
            messagebox.showerror("Σφάλμα!", "Δεν έχουν φορτωθεί νέα δεδομένα για πρόβλεψη.")
            self._log("Σφάλμα: Απαιτούνται νέα δεδομένα για την πρόβλεψη.")
            return

        try:
            # 1. Χρήση εκπαιδευμένου μοντέλου πρόβλεψης
            # 2. Ενημέρωση εσωτερικών flags κατάστασης
            # 3. Εμφάνιση μηνυμάτων (pop-ups) και προβολή μετρικών πρόβλεψης στο log
            self._log("Χρήση του εκπαιδευμένου μοντέλου για πρόβλεψη...")
            self.predictions_df = self.knn_model.predict(self.new_campaign_data, output_path=None) # Κλήση της μεθόδου predict απο το knn_model
            self.predictions_made = True
            self._update_button_states()
            messagebox.showinfo("Πρόβλεψη Ολοκληρώθηκε!", f"Η πρόβλεψη της ανταπόκρισης για τους {len(self.new_campaign_data)} νέους πελάτες ολοκληρώθηκε επιτυχώς.")
            self._log("Η πρόβλεψη ολοκληρώθηκε")
            self._log("\n=================================================\n")
            self._log("Εμφάνιση Τελικών Μετρικών Επικύρωσης.") # Δημιουργία των metrics & εμφάνιση
            self.knn_model.gen_metrics()
            self._log(f"• Mέγεθος test set: {self.knn_model.test_size*100}%")
            self._log(f"• Random state: {self.knn_model.random_state}")
            self._log(self.knn_model.validation_metrics_str)
            self._log("\n=================================================\n")
            messagebox.showinfo("Ειδοποίηση", "Παρακαλώ αποθηκεύστε τα αποτελέσματα της πρόβλεψης.") # Προτροπή χρήστη για αποθήκευση των αποτελεσμάτων πρόβλεψης
            self._log("\nΠαρακαλώ αποθηκεύστε τα αποτελέσματα της πρόβλεψης.")
        except ValueError as ve: # Ανεπαρκή ή λάθος μορφή δεδομένων
            messagebox.showerror("Σφάλμα Πρόβλεψης!", f"Προέκυψε σφάλμα τιμής κατά την πρόβλεψη:\n{str(ve)}")
            self._log(f"Σφάλμα (ValueError) κατά την πρόβλεψη: {str(ve)}")
            self.predictions_df = None # Ακύρωση προβλέψεων
        except Exception as e:
            messagebox.showerror("Σφάλμα Πρόβλεψης!", f"Προέκυψε άγνωστο σφάλμα κατά την πρόβλεψη:\n{str(e)}")
            self._log(f"Άγνωστο σφάλμα κατά την πρόβλεψη: {str(e)}\n")
            self.predictions_df = None # Ακύρωση προβλέψεων

    def quit_app(self, event=None):
        """ 
        Μέθοδος για έξοδο από την εφαρμογή.
        """
        self.master.quit()

    def run(self):
        """
        Μέθοδος που "τρέχει" την εφαρμογή.
        Σύνδεση πλήκτρων q και Esc με την έξοδο της εφαρμογής.
        """
        self.master.bind("<q>", self.quit_app)
        self.master.bind("<Escape>", self.quit_app)
        self.master.mainloop()
