"""
CampaignPredictionApp Module

Παρέχει ένα γραφικό περιβάλλον για:
    - Φόρτωση ιστορικών δεδομένων Excel
    - Εκπαίδευση μοντέλου K-NN (αυτόματα ή χειροκίνητα)
    - Φόρτωση νέων δεδομένων για πρόβλεψη
    - Εκτέλεση προβλέψεων, εμφάνιση μετρικών και γραφημάτων
    - Αποθήκευση αποτελεσμάτων σε αρχείο Excel
    
Usage:
    from gui import CampaignPredictionApp
    app = CampaignPredictionApp(width=1024, height=768)
    app.run()
    
Requirements:
    - Python 3.12.7
    - tkinter 
    - sv_ttk 2.6.0
    - pandas 2.2.3
    - matplotlib 3.10.1
    
Authors:
    Πιτσαρής Κωνσνταντίνος
    Κρανίτσα Αντωνία
    Ραφαήλ Ασλανίδης
"""
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext, ttk, simpledialog
from pathlib import Path
from typing import Optional
import sv_ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model import KNN

class CampaignPredictionApp:
    """
    Γραφικό Περιβάλλον Διεπαφής Χρήστη για προβλέψεις ανταπόκρισης καμπάνιας με
    χρήση Κ-ΝΝ.
    
    Attributes:
        master (tk.Tk): Το κύριο παράθυρο της εφαρμογής.
        past_campaign_data (Optional[pd.DataFrame]): Δεδομένα προηγούμενης καμπάνιας.
        new_campaign_data (Optional[pd.DataFrame]): Δεδομένα νέας καμπάνιας.
        knn_model (Optional[KNN]):Το instance του μοντέλου Κ-ΝΝ.
        predictions_df (Optional[pd.DataFrame]): Τα αποτελέσματα της τελευταίες πρόβλεψης.
        training_data_loaded (bool): Flag φόρτωσης ιστορικών δεδομένων.
        model_trained (bool): Flag ολοκλήρωσης εκπαίδευσης μοντέλου.
        predictions_data_loaded (bool): Flag φόρτωσης νέων δεδομένων.
        predictions_made (bool): Flag ολοκλήρωσης πρόβλεψης.
        
        Οι καταστάσεις της εφαρμογής (βάσει flags) ακολουθούν τη σειρά:
            - Αρχή -> load_past -> training_data_loaded
            - training_data_loaded -> train -> model_trained
            - model_trained -> load_new -> predictions_data_loaded
            - predictions_data_loaded -> predict -> predictions_made
            - predictions_made -> save -> restart (όλα τα flags επανέρχονται)
    """
    def __init__(self, width: int, height: int) -> None:
        """
        Αρχικοποιεί το κύριο παράθυρο της γραφικής διεπαφής και τις βασικές 
        παραμέτρους της εφαρμογής.
        
        Δημιουργεί το παράθυρο της εφαρμογής με συγκεκριμένες διαστάσεις, ορίζει
        τον τίτλο και το θέμα και προετοιμάζει τα απαραίτητα attributes για
        δεδομένα, μοντέλο πρόβλεψης και flags κατάστασης.
        Επιπλέον, καλεί τις μεθόδους για τη δημιουργία των κουμπιών και των
        καρτέλων της διεπαφής.
        
        Args:
            width (int): Το πλάτος του παραθύρου της εφαρμογής σε pixels.
            height (int): Το ύψος του παραθύρου της εφαρμογής σε pixels.
            
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        self.master = tk.Tk()       # Δημιουργία "root" μέσα στη κλάση
        sv_ttk.set_theme('dark')    #Εφαρμογή θέματος sv_ttk
        self.master.title("Εφαρμογή Προβλέψεων Ανταπόκρισης Νέας Καμπάνιας")
        self.master.geometry(f"{width}x{height}")
        self.master.minsize(800,600) # Ελάχιστες διαστάσεις του κεντρικού παραθύρου

        # Αρχικοποίηση των attributes
        self.past_campaign_data = None
        self.new_campaign_data = None
        self.knn_model = None
        self.predictions_df = None

        # Flags για τη διαχείριση κατάστασης κουμπιών
        self.training_data_loaded = False
        self.model_trained = False
        self.predictions_data_loaded = False
        self.predictions_made = False

        # Δημιουργία του περιβάλλοντος διεπαφής
        self._create_buttons()
        self._create_notebook()
        self._log("Ξεκινήστε πρώτα με τη Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας.\n")
        self._update_button_states()

    def _create_buttons(self) -> None:
        """
        Δημιουργεί και τοποθετεί τα βασικά κουμπιά της γραφικής διεπαφής χρήστη.
        
        Η μέθοδος οργανώνει τα κουμπιά μέσα σε πλαίσιο (frame), ρυθμίζει τη διάταξη
        τους και αντιστοιχίζει τις αντίστοιχες εντολές που θα εκτελούνται κατά το 
        πάτημα κάθε κουμπιού.
        Χρησιμοποιείται 'pack()' για γενική διάταξη και 'grid()' όπου 
        απαιτείται ακρίβεια εντός πλαισίων.
        Περιλαμβάνει επιλογές για φόρτωση δεδομένων, εκπαίδευση μοντέλου πρόβλεψης,
        πρόβλεψη και αποθήκευση αποτελεσμάτων.
                    
        Authors:
            Πιτσαρής Κωνσταντίνος
            Κρανίτσα Αντωνία
        """
        button_frame = ttk.Frame(self.master)   # Πλαίσιο για τα κουμπιά
        button_frame.pack(
            padx=10,     # Προσθέτει οριζόντιο χώρο εκτός του πλαισίου του κουμπιού (δεξία-αριστερά)
            pady=10,     # Προσθέτει κατακόρυφο χώρο εκτός του πλαισίου του κουμπιού (πάνω-κάτω)
            fill=tk.X,   # "Γεμίζει" τον οριζόντιο χώρο για να καταλαμβάνει ολό το ελεύθερο πλάτος
            expand=False # Περιορίζει το πλαίσιο των κουμπιών
            )
        # Ρύθμιση για τον τρόπο επέκτασης στηλών των πλαισίων
        for i in range (2):
            button_frame.grid_columnconfigure(i, weight=1)

        # 1. Κουμπί Φόρτωσης Παλαιών Δεδομένων Καμπάνιας
        self.btn_load_past = ttk.Button(
            button_frame,
            text="1. Φόρτωση Δεδομένων Προηγούμενης Καμπάνιας", # Τίτλος κουμπιού
            command=self.load_past_campaign_data # Εντολή που εκτελείται όταν πατιέται το κουμπί
            )
        # 2a. Κουμπί Αυτόματης  Εκπαίδευσης Μοντέλου
        self.btn_train = ttk.Button(
            button_frame, text="2a. Εκπαίδευση Μοντέλου Πρόβλεψης με χρήση βέλτιστου Κ",
            command=self.on_train
            )

        # 2b. Κουμπί Εκπαίδευσης Μοντέλου με είσοδο από χρήστη
        self.btn_manual_train = ttk.Button(
            button_frame, text="2b. Εκπαίδευση Μοντέλου Πρόβλεψης με εισαγωγή K",
            command=self.manual_train
        )

        # 3. Κουμπί Φόρτωσης Νέων Δεδομένων για Πρόβλεψη
        self.btn_load_new = ttk.Button(
            button_frame, text="3. Φόρτωση Δεδομένων Νέας Καμπάνιας",
            command=self.load_new_campaign_data
            )
        # 4. Κουμπί Πρόβλεψης Ανταπόκρισης Νέων Πελατών
        self.btn_predict = ttk.Button(
            button_frame, text="4. Πρόβλεψη Ανταπόκρισης Νέων Πελατών",
            command=self.on_predict
            )
        # 5. Κουμπί Αποθήκευσης Πρόβλεψης
        self.btn_save = ttk.Button(
            button_frame, text="5. Αποθήκευση Πρόβλεψης",
            command=self.save_predictions_wrapper
            )
        # Grid για τη διάταξη των κουμπιών στο πλαίσιο
        for index, button in enumerate([self.btn_load_past,
                                        self.btn_train,
                                        self.btn_manual_train,
                                        self.btn_load_new,
                                        self.btn_predict,
                                        self.btn_save]):
            if button == self.btn_train:
                button.grid(
                    row=index,
                    column=0,
                    columnspan=1,
                    padx=5,
                    pady=5,
                    sticky='nsew'
                    )
            elif button == self.btn_manual_train:
                button.grid(
                    row=index-1,
                    column=1,
                    columnspan=1,
                    padx=5,
                    pady=5,
                    sticky='nsew'
                    )
            else:
                button.grid(
                    row=index,      # Γραμμή
                    column=0,       # Στήλη
                    columnspan=2,   # Χώρος που καταλαμβάνει η στήλη
                    padx=5,         # Οριζόντιο περιθώριο (αριστερά/δεξιά) γύρω από το κουμπί
                    pady=5,         # Κάθετο περιθώριο (πάνω/κάτω) γύρω από το κουμπί
                    sticky='nsew'   # 'Κολλάει' τα κουμπιά στις διαστάσεις του παραθύρου
                    )

    def _create_notebook(self) -> None:
        """        
        Δημιουργεί το notebook της εφαρμογής και τα δύο βασικά tabs.
            
        Το πρώτο tab περιέχει ένα scrolled text widget για την εμφάνιση των
        μηνυμάτων καταγραφής, ενώ το δεύτερο προετοιμάζει ένα καμβά matplotlib
        για την απεικόνιση του γραφήματος πρόβλεψης ανταπόκρισης.
        Χρησιμοποιείται 'pack()' για γενική διάταξη και 'grid()' όπου 
        απαιτείται ακρίβεια εντός πλαισίων.
            
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        # Δημιουργία Notebook για tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
            )

        # Τab #1: Περιοχή κειμένου για εμφάνιση αρχείου καταγραφής
        self.tab_log = ttk.Frame(self.notebook)
        self.text_area = scrolledtext.ScrolledText(
            self.tab_log,
            font=("Trebuchet MS",12),
            )
        self.text_area.pack(
            padx=15,
            pady=15,
            fill=tk.BOTH,
            expand=True
            )
        self.text_area.config(state=tk.DISABLED) # Απενεργοποίηση εγγραφής κειμένου
        self.notebook.add(self.tab_log, text='Αρχείο Καταγραφής')

        # Tab #2: Περιοχή εμφάνισης γραφημάτων (plots)
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text='Γράφημα Πρόβλεψης Ανταπόκρισης')
        self.plot_frame = ttk.Frame(self.tab_plot)
        self.plot_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
            )
        self.fig = plt.figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_button_states(self) -> None:
        """        
        Ενημερώνει τη λειτουργική κατάσταση (ενεργό/ανενεργό) των κουμπιών της 
        εφαρμογής με βάση την τρέχουσα πρόοδο της διαδικασίας πρόβλεψης.
        
        Η μέθοδος βασίζεται σε εσωτερικές boolean μεταβλητές κατάστασης:
            ('training_data_loaded', 'model_trained', 'predictions_data_loaded',
             'predictions_made')
        και ενεργοποιεί μόνο το κουμπί που είναι κατάλληλο για το επόμενο βήμα,
        απενεργοποιώντας τα υπόλοιπα.
        
        Αντίστοιχίες σταδίων:
            - Καμία κατάσταση: ενεργό μόνο το κουμπί φόρτωσης ιστορικών δεδομένων
            - Εκπαίδευση μοντέλου: ενεργά μόνο τα δύο κουμπιά εκπαίδευσης
            - Μοντέλο εκπαιδευμένο: ενεργό μόνο το κουμπί φόρτωσης νέων δεδομένων
            - Νεα δεδομένα φορτωμένα: ενεργό μόνο το κουμπί πρόβλεψης
            - Προβλέψεις ολοκληρώθηκαν: ενεργό μόνο το κουμπί αποθήκευσης
        
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        active, inactive = 'normal', 'disabled'
        states = []
        # Στάδιο 1 - Φόρτωση Ιστορικών Δεδομένων
        if not self.training_data_loaded and not self.model_trained and not self.predictions_data_loaded and not self.predictions_made:
            states = [active, inactive, inactive, inactive, inactive, inactive]
        # Στάδιο 2 - Εκπαίδευση Μοντέλου Πρόβλεψης
        elif self.training_data_loaded and not self.model_trained:
            states = [inactive, active, active, inactive, inactive, inactive]
        # Στάδιο 3 - Φόρτωση Δεδομένων για την πρόβλεψη
        elif self.model_trained and not self.predictions_data_loaded:
            states = [inactive, inactive, inactive, active, inactive, inactive]
        # Στάδιο 4 - Πρόβλεψη Ανταπόκρισης
        elif self.predictions_data_loaded and not self.predictions_made:
            states = [inactive, inactive, inactive, inactive, active, inactive]
        # Στάδιο 5 - Αποθήκευση αποτελεσμάτων
        elif self.predictions_made:
            states = [inactive, inactive, inactive, inactive, inactive, active]
        for button, state in zip([self.btn_load_past, self.btn_train,
                                  self.btn_manual_train, self.btn_load_new,
                                  self.btn_predict, self.btn_save], states):
            button.config(state=state)

    def _log(self, message:str) -> None:
        """
        Καταγράφει ένα μήνυμα στην καρτέλα 'Αρχείο Καταγραφής' της εφαρμογής.
        
        Η μέθοδος ενεργοποιεί προσωρινά την επεξεργασία του πεδίου κειμένου,
        εισάγει το μήνυμα στο τέλος της περιοχής, μετακινεί την προβολή στην πιο
        πρόσφατη εγγραφή και επαναφέρει την περιοχή σε κατάσταση μόνο-ανάγνωσης.
        
        Args:
            message (str): Το μήνυμα που θα προστεθεί στο αρχείο καταγραφής.
            
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        self.notebook.select(self.tab_log)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def _load_data(self, title:str) -> Optional[pd.DataFrame]:
        """
        Φορτώνει δεδομένα από αρχείο Excel μέσω διαλόγου αρχείων και επιστρέφει
        DataFrame.
        
        Ο διάλογος ανοίγματος αρχείου ξεκινάει από τον υποφάκελο 'data',
        παράλληλο με τον φάκελο του κώδικα. Αν δεν υπάρχει, η αρχική τοποθεσία
        γίνεται το τρέχον working directory. Το αρχείο Excel πρέπει υποχρεωτικά
        να περιέχει τις στήλες:
            
            Ηλικία, Φύλο, Περιοχή, Email, Χρήση Κινητού,
            Logins τις τελευταίες 4 εβδομάδες, Logins τους τελευταίους 6 μήνες,
            Αγορές τις τελευταίες 4 εβδομάδες, Αγορές τους τελευταίους 6 μήνες,
            Σύνολο Αγορών, Ανταπόκριση
            
        Αν το DataFrame έχει RangeIndex, αυτό αντικαθίσταται απο index της
        μορφής 'Πελάτης Ν'.
        
        Args:
            title (str): Τίτλος του παραθύρου επιλογής αρχείου.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame με index 'Πελάτης Ν', αλλιώς 'None'

        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        # Βρίσκουμε το κατάλογο του τρέχοντος αρχείου και τον γονέα του γονέα του
        base_dir = Path(__file__).resolve().parent.parent
        # Ορίζουμε τον υποφάκελο ~/data ως default
        default_dir = base_dir / 'data'
        # Εάν δεν υπάρχει - πάμε στο current working directory
        if not default_dir.exists():
            default_dir = Path.cwd()
        file_path= filedialog.askopenfilename(
            title=title,
            initialdir=default_dir,
            filetypes=[("Excel files", "*.xlsx")]
            )
        if not file_path:
            messagebox.showwarning("Προσοχή!", "Δεν επιλέχθηκε αρχείο.")
            return None
        try:
            df = pd.read_excel(file_path)
            
            required_columns = ['Ηλικία', 'Φύλο', 'Περιοχή', 'Email', 'Χρήση Κινητού', 'Logins τις τελευταίες 4 εβδομάδες', 'Logins τους τελευταίους 6 μήνες', 'Αγορές τις τελευταίες 4 εβδομάδες', 'Αγορές τους τελευταίους 6 μήνες', 'Σύνολο Αγορών', 'Ανταπόκριση']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                messagebox.showerror(
                    'Σφάλμα Μορφής Αρχείου',
                    f"Το αρχείο δεν έχει τις στήλες: {', '.join(missing_columns)}"
                    )
                return None
            
            # Έλεγχος εάν το DataFrame είναι κενό
            if df.empty:
                messagebox.showerror(
                    "Σφάλμα!",
                    f"Το αρχείο είναι άδειο ή μη έγκυρου τύπου: {file_path}"
                    )
                return None
            # Δημιουργία δυναμικού λεξικού ανάλογα με το μέγεθος του df
            new_index = {i: f"Πελάτης {i+1}" for i in range(len(df))}
            # Εάν το df έχει το default index, το μετονομάζει
            if isinstance(df.index, pd.RangeIndex):
                df = df.rename(index=new_index)
            # Επιστρέφουμε το DataFrame με νέο index 'Πελάτης N'
            return df
        except FileNotFoundError:
            messagebox.showerror(
                "Σφάλμα!", f"Το αρχείο δεν βρέθηκε: {file_path}"
                )
            return None
        except pd.errors.EmptyDataError:
            messagebox.showerror(
                "Σφάλμα!", f"Το αρχείο είναι εντελώς άδειο: {file_path}"
                )
            return None
        except pd.errors.ParserError:
            messagebox.showerror(
                "Σφάλμα!", f"Σφάλμα τιμής κατά την ανάγνωση του αρχείου Excel: {file_path}"
                )
            return None
        except ValueError as ve:
            messagebox.showerror(
                "Σφάλμα!", f"Σφάλμα τιμής κατά την ανάγνωση του Excel: {file_path}\n{str(ve)}"
                )
            return None
        except Exception as e:
            messagebox.showerror(
                "Σφάλμα!",
                f"Προέκυψε άγνωστο σφάλμα κατά τη φόρτωση του αρχείου Excel: {file_path}\n{str(e)}"
                )
            return None

    def load_past_campaign_data(self) -> None:
        """
        Φορτώνει δεδομένα από αρχείο Excel που αντιστοιχούν σε προηγούμενη
        καμπάνια.
        
        Η μέθοδος καλεί εσωτερικά τον διάλογο επιλογής αρχείου για παλαιά καμπάνια
        Αν η φόρτωση είναι επιτυχής, ενημερώνει τις σχετικές μεταβλητές και 
        flags, μηδενίζει παλαιές προβλέψεις και το εκπαιδευμένο μοντέλο (αν υπάρχουν),
        ενημερώνει κατάλληλα το γραφικό περιβάλλον και τα κουμπιά δράσης και εμφανίζει
        πληροφορίες προς τον χρήστη για το επόμενο βήμα.
        
        Σε περίπτωση αποτυχίας ή ακύρωση της φόρτωσης, καταγράφεται σχετικό μήνυμα
        στο αρχείο καταγραφής της εφαρμογής.
        
        Authors:
            Πιτσαρής Κωνσταντίνος
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
            messagebox.showinfo(
                "Επιτυχία!",
                "Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς."
                )
            self._log("Τα δεδομένα της προηγούμενης καμπάνιας φορτώθηκαν επιτυχώς.")
            messagebox.showinfo("Επόμενο Βήμα","Προχωρήστε στην Εκπαίδευση Μοντέλου Πρόβλεψης.")
            messagebox.showwarning(
                "Προειδοποίηση!",
                "Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα."
                )
            self._log("\nΕπόμενο βήμα: Προχωρήστε στην εκπαίδευση του μοντέλου πρόβλεψης.")
            self._log("Προειδοποίηση: Αυτή η διαδικασία ενδέχεται να διαρκέσει περισσότερη ώρα.")
            self._log(
                "Το γραφικό περιβάλλον της εφαρμογής πιθανώς να 'παγώσει' κατά τη διάρκεια της εκπαίδευσης."
                )
        else:
            self._log("Η φόρτωση των παλαιών δεδομένων απέτυχε ή ακυρώθηκε.")

    def load_new_campaign_data(self) -> None:
        """
        Φορτώνει δεδομένα από αρχείο Excel που αντιστοιχούν σε νέα καμπάνια.

        Η μέθοδος καλεί εσωτερικά τον διάλογο επιλογής αρχείου για τη νέα καμπάνια.
        Αν η φόρτωση είναι επιτυχής, ενημερώνει τις σχετικές μεταβλητές και
        flags, μηδενίζει παλαιές προβλέψεις (αν υπάρχουν), ενημερώνει το γραφικό
        περιβάλλον και εμφανίζει πληροφορίες προς τον χρήστη για το επόμενο βήμα.
        
        Σε περίπτωση αποτυχίας ή ακύρωση της φόρτωσης, καταγράφεται σχετικό μήνυμα
        στο αρχείο καταγραφής της εφαρμογής.
        
        Authors:
            Πιτσαρής Κωνσταντίνος
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
            messagebox.showinfo(
                "Επόμενο Βήμα",
                "Προχωρήστε στην Πρόβλεψη Ανταπόκρισης Νέων Πελατών."
                )
            self._log("Επόμενο βήμα: Προχωρήστε στην πρόβλεψη ανταπόκρισης νέων πελατών.")
        else:
            self._log("Η φόρτωση των νέων δεδομένων απέτυχε ή ακυρώθηκε.")
            self._log("\n=================================================\n")

    def _save_predictions(self, df_to_save: Optional[pd.DataFrame]) -> bool:
        """
        Αποθηκεύει το DataFrame πρόβλεψης σε αρχείο Excel μέσω διαλόγου αποθήκευσης.
        
        Ελέγχει αν το DataFrame είναι κενό ή None, ανοίγει διάλογο για επιλογή
        διαδρομής αρχείου Excel και γράφει τα δεδομένα. Σε περίπτωση σφάλματος
        κατά την αποθήκευση, εμφανίζει μήνυμα και καταγράφει το σφάλμα στο 
        αρχείο καταγραφής της εφαρμογής.
        
        Args:
            df_to_save (Optional[pd.DataFrame]): Το DataFrame με τις προβλέψεις
            προς αποθήκευση.
            
        Returns:
            bool: 
                - True: αν η αποθήκευση ολοκληρώθηκε επιτυχώς.
                - False: ακύρωση αποθήκευσης ή σφάλμα.
                
        Authors:
            Πιτσαρής Κωνσταντίνος
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
            messagebox.showinfo(
                "Επιτυχία",
                f"Οι προβλέψεις αποθηκεύτηκαν με επιτυχία στο:\n{save_path}"
                )
            return True
        except Exception as e:
            messagebox.showerror(
                "Σφάλμα Αποθήκευσης!",
                f"Δεν ήταν δυνατή η αποθήκευση των προβλέψεων.\nΣφάλμα:{str(e)}"
                )
            self._log(f"Σφάλμα κατά την αποθήκευση στο {save_path}: {str(e)}")
            return False

    def save_predictions_wrapper(self) -> None:
        """
        Διαχειρίζεται τη ροή αποθήκευσης προβλέψεων και επαναφέριε την εφαρμογή
        σε αρχική κατάσταση.
        
        Καλεί την εσωτερική μέθοδο αποθήκευσης αν υπάρχουν προβλέψεις και σε 
        επιτυχία μηδενίζει όλα τα δεδομένα και flags, ενημερώνει καταλλήλως τα
        κουμπιά και το αρχείο καταγραφής της εφαρμογής και ειδοποιεί τον χρήστη
        ότι μπορεί να ξεκινήσει εκ νέου με νέες προβλέψεις.
        Σε περίπτωση έλλεψιςη προβλέψεων, εμφανίζει προειδοποίηση.
        
        Authors:
            Πιτσαρής Κωνσταντίνος        
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
                self._log("Επαναφορά εφαρμογής σε αρχική κατάσταση.")
                self._log("\n===============ΤΕΛΟΣ ΠΡΟΓΡΑΜΜΑΤΟΣ===============")
                self._log("\n=== Το πρόγραμμα επανήλθε στην αρχική κατάσταση. ===")
        else:
            messagebox.showwarning("Προσοχή!", "Δεν έχουν δημιουργηθεί προβλέψεις προς αποθήκευση.")
            self._log("Αποτυχία αποθήκευσης: Δεν υπάρχουν διαθέσιμες προβλέψεις.")

    def on_train(self) -> None:
        """
        Ξεκινά την αυτόματα εκπαίδευση του μοντέλου K-NN με βελτιστοποίηση του k.
        
        Ελέγχει πρώτα ότι έχουν φορτωθεί δεδομένα εκπαίδευσης. Έπειτα:
            - Αρχικοποιεί το K-NN μοντέλο
            - Τροφοδοτεί τα δεδομένα εκπαίδευσης
            - Εντοπίζει το βέλτιστο k μέσω cross-validation
            - Εκπαιδεύει το τελικό μοντέλο με το βέλτιστο k
        Σε περίπτωση σφάλματος, επαναφέρει την κατάσταση και ενημερώνει το γραφικό
        περιβάλλον διεπαφής.
        
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
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
            k_range = range(2, 16) # Μεγαλύτερο εύρος γειτόνων = αργότερη εκτέλεση
            fold_range = range(2, 8) # Μεγαλύτερο εύρος folds = αργότερη εκτέλεση
            self.knn_model.find_best_neighbors(k_range=k_range, fold_range=fold_range)
            self.model_trained = True
            self._update_button_states()
            self._log("Ολοκληρώθηκε η διαδικασία εύρεσης βέλτιστου αριθμού γειτόνων (k)")
            self._log(f"\n    -Βέλτιστος αριθμός γειτόνων (k):{self.knn_model.best_n_neighbors}\n")
            self._log("Εκπαίδευση τελικού μοντέλου με τα πλήρη δεδομένα εκπαίδευσης...")
            self.knn_model.fit()
            self._log(
                "\n   -Το τελικό μοντέλο εκπαιδεύτηκε με k = " 
                + str(self.knn_model.best_n_neighbors)
                )
            messagebox.showinfo(
                "Εκπαίδευση Ολοκληρώθηκε!",
                f"Βέλτιστο k = {self.knn_model.best_n_neighbors}\n"
                )
            self._log("\nΕμφάνιση Μετρικών Επικύρωσης.")
            # Δημιουργία των metrics & εμφάνιση
            self.knn_model.gen_metrics()
            self._log(f"• Mέγεθος test set: {self.knn_model.test_size*100}%")
            self._log(f"• Random state: {self.knn_model.random_state}")
            self._log(self.knn_model.validation_metrics_str)
            self._log("\n=================================================\n")
            self._log("\n=================================================\n")
            # Ακύρωση τυχόν προηγούμενων προβλέψεων τώρα που το μοντέλο επανεκπαιδεύτηκε
            self.predictions_df = None
            messagebox.showinfo(
                "Επόμενο Βήμα", "Προχωρήστε στη φόρτωση των δεδομένων νέας καμπάνιας."
                )
            self._log("Επόμενο βήμα: Προχωρήστε στη φόρτωση των δεδομένων νέας καμπάνιας.")
        except ValueError as ve: # Ανεπαρκή ή λάθος μορφή δεδομένων
            messagebox.showerror(
                "Σφάλμα Εκπαίδευσης!",
                f"Προέκυψε σφάλμα τιμής κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(ve)}"
                )
            self._log(f"Σφάλμα (ValueError) κατά την εκπαίδευση: {str(ve)}")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
            self.model_trained = False
            self.training_data_loaded = True
            self._update_button_states()
        except Exception as e:
            messagebox.showerror(
                "Σφάλμα Εκπαίδευσης!",
                f"Προέκυψε άγνωστο σφάλμα κατά την εκπαίδευση του μοντέλου πρόβλεψης:\n{str(e)}"
                )
            self._log(f"Άγνωστο σφάλμα κατά την εκπαίδευση: {str(e)}\n")
            self.knn_model = None # Ακύρωση επεξεργαστή σε περίπτωση σφάλματος
            self.model_trained = False
            self.training_data_loaded = True
            self._update_button_states()

    def manual_train(self) -> None:
        """
        Εκτελεί χειροκίνητη εκπαίδευση του μοντέλου K-NN με τιμή k που εισάγει
        ο χρήστης.
        
        Η μέθοδος ζητάει από τον χρήστη έναν ακέραιο k μέσω διαλόγου, δημιουργεί
        νέο KNN αντικείμενο με τον δοσμένο k, τροφοδοτεί τα δεδομένα εκπαίδευσης
        και εκπαιδεύει το μοντέλο πρόβλεψης. Ενημερώνει τα flags κατάστασης και
        τα κουμπιά της διεπαφής ανάλογα με την έκβαση.
        
        Authors:
            Κρανίτσα Αντωνία
        """
        self._log("\n=== Χειροκίνητη Εκπαίδευση Μοντέλου ===\n")

        if self.past_campaign_data is None:
            messagebox.showerror(
                "Σφάλμα Εκπαίδευσης!",
                "Προέκυψε σφάλμα φόρτωσης δεδομένων κατά την εκπαίδευση του μοντέλου πρόβλεψης."
                )
            self._log(
                "Σφάλμα Εκπαίδευσης! " 
                + "- Προέκυψε σφάλμα φόρτωσης δεδομένων κατά την εκπαίδευση του μοντέλου πρόβλεψης."
                )
            return

        try:
            k = simpledialog.askinteger(
                "Εισαγωγή K",
                "Εισάγετε τον αριθμό γειτόνων (k) για το μοντέλο KNN:",
                minvalue=1,
                maxvalue=100
            )
            if k is None:
                self._log("Η χειροκίνητη εκπαίδευση ακυρώθηκε από τον χρήστη.")
                self.model_trained = False
                self._update_button_states()
                return

            self._log(f"Εκπαίδευση μοντέλου με K = {k} γείτονες...")
            self.knn_model = KNN(neighbors=k, test_size=0.2, random_state=42)
            self.knn_model.feed_data(self.past_campaign_data)
            self.knn_model.fit()

            self.model_trained = True
            self.predictions_df = None
            self._update_button_states()

            self._log("Εκπαίδευση μοντέλου με K = " + str(k) + " ολοκληρώθηκε.")
            self._log("\nΕμφάνιση  Μετρικών Επικύρωσης.")
            # Δημιουργία των metrics & εμφάνιση
            self.knn_model.gen_metrics()
            self._log(f"• Mέγεθος test set: {self.knn_model.test_size*100}%")
            self._log(f"• Random state: {self.knn_model.random_state}")
            self._log(self.knn_model.validation_metrics_str)
            self._log("\n=================================================\n")
            messagebox.showinfo(
                "Εκπαίδευση Ολοκληρώθηκε!", f"To μοντέλο εκπαιδεύτηκε επιτυχώς με k = {k}"
                )
            self._log("Επόμενο βήμα: Προχωρήστε στη φόρτωση των δεδομένων νέας καμπάνιας.\n")

        except ValueError as ve:
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Σφάλμα κατά την εκπαίδευση:\n{str(ve)}")
            self._log(f"Σφάλμα (ValueError) κατά την εκπαίδευση: {str(ve)}")
            self.knn_model = None
            self.model_trained = False
            self._update_button_states()
        except Exception as e:
            messagebox.showerror("Σφάλμα Εκπαίδευσης!", f"Άγνωστο σφάλμα:\n{str(e)}")
            self._log(f"Άγνωστο σφάλμα κατά την εκπαίδευση: {str(e)}")
            self.knn_model = None
            self.model_trained = False
            self._update_button_states()

    def on_predict(self) -> None:
        """
        Εκκινεί τη διαδικασία πρόβλεψης ανταπόκρισης για τη νέα καμπάνια.
        
        Η μέθοδος ελέγχει πρώτα ότι έχουν φορτωθεί το εκπαιδευμένο μοντέλο και
        τα δεδομένα της νέας καμπάνιας. Στη συνέχεια καλεί το μοντέλο για να 
        προβλέψει την ανταπόκριση, ενημερώνει τα εσωτερικά flags και τα κουμπιά, 
        εμφανίζει στο χρήστη μηνύματα επιτυχίας και τα μετρικά ελέγχου και 
        αποτυπώνει όλα τα στατιστικά στο αρχείο καταγραφής. Τέλος, δείχνει
        διάγραμμα πίτας με την κατανομή των απαντήσεων ανά φύλο.
        
        Authors:
            Πιτσαρής Κωνσταντίνος
        """
        self._log("\n=== Έναρξη Διαδικασίας Πρόβλεψης ===\n")
        if self.knn_model is None: # Έλεγχος εάν υπάρχει διαθέσιμο εκπαιδευμένο μοντέλο πρόβλεψης
            messagebox.showerror(
                "Σφάλμα!",
                "Δεν υπάρχει εκπαιδευμένο μοντέλο πρόβλεψης.\nΕκτελέστε πρώτα την εκπαίδευση."
                )
            self._log("Σφάλμα: Απαιτείται εκπαιδευμένο μοντέλο για την πρόβλεψη.")
            return

        if self.new_campaign_data is None:  # Έλεγχος εάν έχουν φορτωθεί νέα δεδομένα για πρόβλεψη
            messagebox.showerror("Σφάλμα!", "Δεν έχουν φορτωθεί νέα δεδομένα για πρόβλεψη.")
            self._log("Σφάλμα: Απαιτούνται νέα δεδομένα για την πρόβλεψη.")
            return

        try:
            self._log("Χρήση του εκπαιδευμένου μοντέλου για πρόβλεψη...")
            # Κλήση της μεθόδου predict απο το knn_model
            self.predictions_df = self.knn_model.predict(self.new_campaign_data, output_path=None)
            self.predictions_made = True
            self._update_button_states()
            messagebox.showinfo(
                "Πρόβλεψη Ολοκληρώθηκε!",
                f"Η πρόβλεψη της ανταπόκρισης για τους {len(self.new_campaign_data)} νέους πελάτες ολοκληρώθηκε επιτυχώς."
                )
            self._log("Η πρόβλεψη ολοκληρώθηκε")
            self._log("\n=================================================\n")
            # Προτροπή χρήστη για αποθήκευση των αποτελεσμάτων πρόβλεψης
            messagebox.showinfo("Ειδοποίηση", "Παρακαλώ αποθηκεύστε τα αποτελέσματα της πρόβλεψης.")
            # Εμφάνιση γραφήματος πίτας ανταπόκρισης (gender / yes-no)
            self.responses_by_gender_pie()
            self._log("\nΠαρακαλώ αποθηκεύστε τα αποτελέσματα της πρόβλεψης.")
        except ValueError as ve: # Ανεπαρκή ή λάθος μορφή δεδομένων
            messagebox.showerror(
                "Σφάλμα Πρόβλεψης!", f"Προέκυψε σφάλμα τιμής κατά την πρόβλεψη:\n{str(ve)}"
                )
            self._log(f"Σφάλμα (ValueError) κατά την πρόβλεψη: {str(ve)}")
            self.predictions_df = None # Ακύρωση προβλέψεων
            self._update_button_states()
        except Exception as e:
            messagebox.showerror(
                "Σφάλμα Πρόβλεψης!", f"Προέκυψε άγνωστο σφάλμα κατά την πρόβλεψη:\n{str(e)}"
                )
            self._log(f"Άγνωστο σφάλμα κατά την πρόβλεψη: {str(e)}\n")
            self.predictions_df = None # Ακύρωση προβλέψεων
            self._update_button_states()

    def responses_by_gender_pie(self) -> None:
        """
        Δημιουργεί και εμφανίζει ένα διάγραμμα πίτας που απεικονίζει τις απαντήσεις
        χωριστά άνα φύλο.
        
        Η μέθοδος βασίζεται στα δεδομένα προβλέψεων. Εάν δεν υπάρχουν δεδομένα
        προβλέψεων ('self.predictions_df' είναι None), εμφανίζει μήνυμα λάθους
        και καταγράφει σχετικό μήνυμα στο αρχείο καταγραφής.
        
        Η μέθοδος μετατρέπει τα ονόματα στηλών σε μικρά γράμματα χωρίς κενά και
        μετονομάζει τις στήλες 'Φύλο' και 'Ανταπόκριση' σε 'gender' και 'response'
        αντίστοιχα, ώστε να υπολογίσει το ποσοστό των 'yes' απαντήσεων ανά φύλο.
        
        Στη συνέχεια σχεδιάζει δύο διαγράμματα πίτας (ένα για κάθε φύλο) με
        διαφορετικά χρώματα και τίτλους.
        
        Authors:
            Κρανίτσα Αντωνία
        """
        if self.predictions_df is None:
            self._log('Δεν υπάρχουν στοιχεία πρόβλεψης για δημιουργία γραφήματος.')
            messagebox.showerror(
                'Σφάλμα', 'Δεν υπάρχουν στοιχεία πρόβλεψης για δημιουργία γραφήματος.'
                )
            return
        df = self.predictions_df.copy()
        # Μετατροπή ονομάτων στηλών
        df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]
        df.rename(columns={'φύλο': 'gender', 'ανταπόκριση': 'response'}, inplace=True)
        grouped = df.groupby('gender')['response'].value_counts().unstack().fillna(0)
        grouped['percentage_yes'] = (grouped.get('yes', 0) / grouped.sum(axis=1)) * 100
        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        for i, gender in enumerate(grouped.index):
            yes = grouped.loc[gender, 'yes'] if 'yes' in grouped.columns else 0
            no = grouped.loc[gender, 'no'] if 'no' in grouped.columns else 0
            colors = ['#ADD8E6', '#9400D3']
            axes[i].pie(
                [yes, no],
                labels=['Yes', 'No'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors)
            axes[i].set_title(f'Responses for {gender.capitalize()}')
        self.canvas.draw()

    def quit_app(self, event: Optional[tk.Event] = None) -> None:
        """ 
        Κλείνει την εφαρμογή.
        
        Args:
            event (Optional[tk.Event], optional): Το γεγονός που ενεργοποιεί την
            εξόδο από την εφαρμογή (π.χ πάτημα πλήκτρου).
                Προεπιλεγμένη τιμή είναι None.
                
       Authors:
            Ασλανίδης Ραφαήλ
        """
        self.master.quit()

    def run(self) -> None:
        """
        Εκκινεί την εφαρμογή και συνδέει πλήκτρα εξόδου.
        
        Συνδέει τα πλήκτρα 'q' και 'Esc' με την έξοδο της εφαρμογής και ξεκινά
        τον κύριο βρόχο του γραφικού περιβάλλοντος.
        
        Authors:
            Ασλανίδης Ραφαήλ
            Πιτσαρής Κωνσταντίνος
        """
        self.master.bind("<q>", self.quit_app)
        self.master.bind("<Escape>", self.quit_app)
        self.master.mainloop()
