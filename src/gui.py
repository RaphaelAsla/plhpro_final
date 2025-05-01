import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import pandas as pd
from model import KNN

class CampaignPredictionApp:
    """ Κλάση υλοποίησης Γραφικού Περιβάλλοντος Εφαρμογής Προβλέψεων Ανταπόκρισης
        Πελατών με χρήση μοντέλου μηχανικής εκμάθησης K-nn"""
    def __init__(self, width, height):
        # Αρχικοποίηση της εφαρμογής GUI
        self.master = tk.Tk() # Δημιουργία "root" μέσα στη κλάση
        self.master.title("Εφαρμογή Προβλέψεων Ανταπόκρισης Νέας Καμπάνιας") # Τίτλος Παραθύρου Εφαρμογής
        self.master.geometry(f"{width}x{height}") # Διαστάσεις Παραθύρου Εφαρμογής - Παίρνει όρισμα στη main
        self.master.minsize(800,600) # Ελάχιστες διαστάσεις του κεντρικού παραθύρου
        self.past_campaign_data = None # το df με τα στοιχεία της προηγούμενης καμπάνιας
        self.new_campaign_data = None # το df με τα στοιχεία της νέας καμπάνιας
        self.knn_model = None # το instance της Κ-ΝΝ που είσαγεται
        self.predictions_df = None # το df με τα στοιχεία της πρόβλεψης
        self._create_widgets() # Δημιουργία των widgets της εφαρμογής
        messagebox.showinfo("Πληροφορία","Ξεκινήστε με τη Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας.")
        self._log("Ξεκινήστε πρώτα με τη Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας.\n")
        
    def _create_widgets(self):
        """Δημιουργία των στοιχείων του Γραφικού Περιβάλλοντος"""
        button_frame = tk.Frame(self.master)    # Πλαίσιο για τα κουμπιά
        button_frame.pack(
            padx=10, # Προσθέτει οριζόντιο χώρο εκτός του πλαισίου του κουμπιού (δεξία & αριστερά)
            pady=10, # Προσθέτει κατακόρυφο χώρο εκτός του πλαισίου του κουμπιού (πάνω & κάτω)
            fill=tk.X, # "Γεμίζει" όλο τον οριζόντιο χώρο για να καταλαμβάνει ολό το ελεύθερο πλάτος
            expand=False # Περιορίζει το πλαίσιο των κουμπιών για να μην "κλέβουν" χώρο απο την text_area
            )
        button_frame.grid_columnconfigure(0, weight=2)  # ρυθμίζει το πως επεκτείνονται οι στήλες των πλαισίων
        button_frame.grid_columnconfigure(1, weight=2)
        
        btn_load_past = tk.Button(  # Κουμπί Φόρτωσης Παλαιών Δεδομένων Καμπάνιας
            button_frame,
            text="1. Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας", # Τίτλος κουμπιού
            font=("Arial",12), # Γραμματοσειρά
            bg="tan4", # Χρώμα πλαισίου
            fg="white", # Χρώμα γραμματοσειράς
            command=self.load_past_campaign_data # Εντολή που εκτελείται όταν πατιέται το κουμπί
            )
        
        btn_load_past.grid(
            row=0,
            column=0,   
            columnspan=2, # Χώρος που καταλαμβάνει η στήλη
            padx=5, # Τοποθέτηση κουμπιού στον άξονα Χ του παραθύρου
            pady=5, # Τοποθέτηση κουμπιού στον άξονα Υ του παραθύρου
            sticky="nsew" # Κάνει τα κουμπιά να "κολλάνε" στις διαστάσεις του παραθύρου
            )
        
        btn_load_new = tk.Button( # Κουμπί Φόρτωσης Νέων Δεδομένων για Πρόβλεψη
            button_frame,
            text="3. Φόρτωση Δεδομένων Νέας Καμπάνιας",
            font=("Arial",12),
            bg="tan4",
            fg="white",
            command=self.load_new_campaign_data
            )
        
        btn_load_new.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )
        
        btn_train = tk.Button( # Κουμπί Εκπαίδευσης Μοντέλου
            button_frame,
            text="2. Εκπαίδευση Μοντέλου Πρόβλεψης",
            font=("Arial",12),
            bg="royal blue",
            fg="white",
            command=self.on_train
            )
        
        btn_train.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=5,
            pady=10,     
            sticky="nsew"
            )
        
        btn_predict = tk.Button( # Κουμπί Πρόβλεψης Ανταπόκρισης Νέων Πελατών
            button_frame,
            text="4. Πρόβλεψη Ανταπόκρισης Νέων Πελατών",
            font=("Arial",12),
            bg="royal blue",
            fg="white",
            command=self.on_predict
            )
        
        btn_predict.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew"
            )
        
        btn_save_predictions = tk.Button( # Κουμπί Αποθήκευσης Πρόβλεψης
            button_frame,
            text="5. Αποθήκευση Πρόβλεψης",
            font=("Arial", 12),
            bg="sea green",
            fg="white",
            command=self.save_predictions_wrapper
            )
        
        btn_save_predictions.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=5,
            pady=10,
            sticky="nsew"
            )
        
        self.text_area = scrolledtext.ScrolledText( # Περιοχή κειμένου για εμφάνιση αποτελεσμάτων
            self.master,
            width=100,
            height=20,
            font=("Arial",11),
            )
        self.text_area.pack(
            padx=15,
            pady=15,
            fill=tk.BOTH,
            expand=True
            )
        
        self.text_area.config(state=tk.DISABLED) # Απενεργοποίηση δυνατότητας εγγραφής κειμένου
        
    def _log(self, message):
        """Βοηθητική μέθοδος για προσθήκη κειμένου στην περιοχή κειμένου"""        
        self.text_area.config(state=tk.NORMAL) # Ενεργοποίηση δυνατότητας εγγραφής κειμένου
        self.text_area.insert(tk.END, message + "\n") # Εμφανίζει το κείμενο message 
        self.text_area.see(tk.END) # Μετακίνηση στο τέλος του κειμένου
        self.text_area.config(state=tk.DISABLED) # Ξανά απενεργοποίηση δυνατότητας εγγραφής

    def _load_data(self,title):
        """Εσωτερική μέθοδος κλάσης φόρτωσης δεδομένων απο αρχείο Excel + Διαχείριση Exceptions"""        
        file_path= filedialog.askopenfilename(
            title=title,
            filetypes=[("Excel files", "*.xlsx")]
            )
        if not file_path:
            messagebox.showwarning("Προσοχή!", "Δεν επιλέχθηκε αρχείο.")
            return None
        try:
            df = pd.read_excel(file_path)       
            if df.empty: # Έλεγχος εάν το DataFrame είναι κενό
                messagebox.showerror("Σφάλμα!",f"Το αρχείο είναι άδειο ή είναι μη έγκυρου τύπου: {file_path}")
                return None
            new_index = {i: f"Πελάτης {i+1}" for i in range(len(df))} # λεξικό δυναμικό ανάλογα με το μέγεθος του df
            if isinstance(df.index, pd.RangeIndex): # εάν το df έχει το default index, το μεταονομάζει
                df = df.rename(index=new_index)
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
        """Μέθοδος Φόρτωσης Δεδομένων Προηγούμενης Καμπάνιας"""         
        self._log("\n=== Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας ===\n")
        temp_data = self._load_data("Επιλέξτε αρχείο δεδομένων προηγούμενης καμπάνιας")
        if temp_data is not None:
            self.past_campaign_data = temp_data
            messagebox.showinfo("Επιτυχία!", "Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς.")
            messagebox.showinfo("Επόμενο Βήμα","Προχωρήστε στην Εκπαίδευση Μοντέλου Πρόβλεψης.")
            messagebox.showwarning("Προειδοποίηση!", "Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα.")
            self._log("\nΕπόμενο βήμα: Προχωρήστε στην εκπαίδευση του μοντέλου πρόβλεψης.")
            self._log("Προειδοποίηση: Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα.")
            self._log("Το γραφικό περιβάλλον της εφαρμογής πιθανώς να 'παγώσει' κατά τη διάρκεια της εκπαίδευσης.")
            self.knn_model = None # Ακύρωση τυχόν υπάρχοντος εκπαιδευμένου μοντέλου
            self.predictions_df = None # Ακύρωση τυχόν προβλέψεων
        else:
            self._log("Η φόρτωση των παλαιών δεδομένων απέτυχε ή ακυρώθηκε.")
        
    def load_new_campaign_data(self):
        """Μέθοδος Φόρτωσης Δεδομένων της Νέας Καμπάνιας"""        
        self._log("\n=== Φόρτωση Δεδομένων Νέας Καμπάνιας ===\n")
        temp_data = self._load_data("Επιλέξτε αρχείο δεδομένων νέας καμπάνιας")
        if temp_data is not None:
            self.new_campaign_data = temp_data
            messagebox.showinfo("Επιτυχία!", "Τα δεδομένα της νέας καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("Τα δεδομένα της νέας καμπάνιας φορτώθηκαν επιτυχώς.")
            self._log("\n=================================================\n")
            messagebox.showinfo("Επόμενο Βήμα", "Προχωρήστε στην Πρόβλεψη Ανταπόκρισης Νέων Πελατών.")
            self._log("Επόμενο βήμα: Προχωρήστε στην πρόβλεψη ανταπόκρισης νέων πελατών.")
            self.predictions_df = None # Ακύρωση προβλέψεων εάν φορτωθούν νέα δεδομένα
        else:
            self._log("Η φόρτωση των νέων δεδομένων απέτυχε ή ακυρώθηκε.")
            self._log("\n=================================================\n")
        
    def _save_predictions(self, df_to_save):
        """ Εσωτερική Μέθοδος για την Αποθήκευση των Προβλέψεων σε νέο αρχείο Excel
            Παίρνει το dataframe προς αποθήκευση ως όρισμα """            
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
        """Βοηθητική Μέθοδος που καλείται απο το Κουμπί Αποθήκευσης"""        
        self._log("\n=== Aποθήκευση Προβλέψεων ===")
        if self.predictions_df is not None:
            self._save_predictions(self.predictions_df)
            self._log("\n===============ΤΕΛΟΣ ΠΡΟΓΡΑΜΜΑΤΟΣ===============")
        else:
            messagebox.showwarning("Προσοχή!", "Δεν έχουν δημιουργηθεί προβλέψεις προς αποθήκευση.")
            self._log("Αποτυχία αποθήκευσης: Δεν υπάρχουν διαθέσιμες προβλέψεις.")
                    
    def on_train(self):
        """Μέθοδος που εκτελείται όταν πατηθεί το κουμπί Εκπαίδευσης"""        
        self._log("\n=== Έναρξη Διαδικασίας Εκπαίδευσης Μοντέλου Πρόβλεψης Κ-nn ===\n")
        if self.past_campaign_data is None: # Έλεγχος εάν έχουν φορτωθεί δεδομένα εκπαίδευσης
            messagebox.showerror("Σφάλμα!", "Δεν έχουν φορτωθεί δεδομένα εκπαίδευσης.")
            self._log("Σφάλμα: Απαιτούνται δεδομένα εκπαίδευσης.")
            return
        try:
            self._log("Aρχικοποίηση επεξεργαστή K-nn...")
            self.knn_model = KNN(neighbors=None, test_size=0.2, random_state=42)
            self._log("Τροφοδότηση δεδομένων εκπαίδευσης στο μοντέλο...")
            self.knn_model.feed_data(self.past_campaign_data)
            
            self._log("Εύρεση βέλτιστου αριθμού γειτόνων (k)...")

            k_range = range(2, 16) # Προσαρμόζουμε το εύρος ανάλογα (μεγαλύτερο εύρος γειτόνων=αργότερη εκτέλεση)
            fold_range = range(2, 8) # Προσαρμόζουμε το εύρος ανάλογα (μεγαλύτερο εύρος folds=αργότερη εκτέλεση)
            self.knn_model.find_best_neighbors(k_range=k_range, fold_range=fold_range)
                    
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
        except ValueError as ve:
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Προέκυψε σφάλμα τιμής κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(ve)}")
            self._log(f"Σφάλμα (ValueError) κατά την εκπαίδευση: {str(ve)}")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
        except Exception as e:
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Προέκυψε άγνωστο σφάλμα κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(e)}")
            self._log(f"Άγνωστο σφάλμα κατά την εκπαίδευση: {str(e)}\n")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
            
    def on_predict(self):
        """Μέθοδος που εκτελείται όταν πατιέται το Κουμπί Πρόβλεψης"""        
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
            self._log("Χρήση του εκπαιδευμένου μοντέλου για πρόβλεψη...")

            self.predictions_df = self.knn_model.predict(self.new_campaign_data, output_path=None) # Κλήση της μεθόδου predict απο το knn_model

            messagebox.showinfo("Πρόβλεψη Ολοκληρώθηκε!", f"Η πρόβλεψη της ανταπόκρισης για τους {len(self.new_campaign_data)} νέους πελάτες ολοκληρώθηκε επιτυχώς.\nΜπορείτε να αποθηκεύσετε τα αποτελέσματα με το κουμπί 'Αποθήκευση Πρόβλεψης'.")
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
        except ValueError as ve:
            messagebox.showerror("Σφάλμα Πρόβλεψης!", f"Προέκυψε σφάλμα τιμής κατά την πρόβλεψη:\n{str(ve)}")
            self._log(f"Σφάλμα (ValueError) κατά την πρόβλεψη: {str(ve)}")
            self.predictions_df = None # Ακύρωση προβλέψεων
        except Exception as e:
            messagebox.showerror("Σφάλμα Πρόβλεψης!", f"Προέκυψε άγνωστο σφάλμα κατά την πρόβλεψη:\n{str(e)}")
            self._log(f"Άγνωστο σφάλμα κατά την πρόβλεψη: {str(e)}\n")
            self.predictions_df = None # Ακύρωση προβλέψεων

    def quit_app(self, event=None):
        self.master.quit()
                             
    def run(self):
        """Μέθοδος που "τρέχει" την εφαρμογή"""        
        self.master.bind("<q>", self.quit_app)
        self.master.bind("<Escape>", self.quit_app)
        self.master.mainloop()