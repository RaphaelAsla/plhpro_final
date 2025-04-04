# Import απαραίτητων βιβλιοθηκών
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import os
import pandas as pd

# Βοηθητικό λεξικό για την αλλάγη ονόματος στα index των dataframes
new_index = {}
for i in range(1000):
    new_index[i] = f'Πελάτης {i+1}'

# Συνάρτηση για φόρτωση δεδομένων
def load_data (title):
    file_path = filedialog.askopenfilename(title=title, filetypes=[('Excel files', '*.xlsx')])
    if not file_path:
        messagebox.showerror('Σφάλμα', 'Δεν έχει επιλεγεί αρχείο!')
    try:
        df = pd.read_excel(file_path)
        df = df.rename(index = new_index)
        return df
    except FileNotFoundError:
        messagebox.showerror('Σφάλμα', f'Το αρχείο δεν βρέθηκε: {file_path}')
        return None
    except pd.errors.EmptyDataError:
        messagebox.showerror('Σφάλμα', f'Το αρχείο είναι άδειο: {file_path}')
        return None
    except pd.errors.ParserError:
        messagebox.showerror('Σφάλμα', f'Σφάλμα κατά την ανάγωνση του αρχείου Excel: {file_path}')
        return None
    except Exception as e:
        messagebox.showerror('Σφάλμα', f'Προέκυψε σφάλμα κατά τη φόρτωση του αρχείου {str(e)}.')
        return None

# Συνάρτηση για φόρτωση ιστορικών δεδομένων
def load_past_campaign_data():
    past_campaign_data = load_data("Επιλέξτε αρχείο δεδομένων προηγούμενης καμπάνιας")
    if past_campaign_data is None:
        messagebox.showerror("Σφάλμα", "Τα δεδομένα της προηγούμενης καμπάνιας δεν φορτώθηκαν.")
        return None
    return past_campaign_data

# Συνάρτηση για φόρτωση νέων δεδομένων
def load_new_campaign_data():
    new_campaign_data = load_data("Επιλέξτε αρχείο δεδομένων νέας καμπάνιας")
    if new_campaign_data is None:
        messagebox.showerror("Σφάλμα", "Τα δεδομένα της νέας καμπάνιας δεν φορτώθηκαν.")
        return None
    return new_campaign_data

# Συνάρτηση για την Αποθήκευση των προβλέψεων σε νέο αρχείο Excel
def save_predictions(df):
    messagebox.showinfo('Ενημέρωση','Αποθήκευση προβλεπόμενης ανταπόκρισης πελατών νέας καμπάνιας.')
    try:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if os.path.exists(save_path):
            overwrite = messagebox.askyesno("Προσοχή", "Θέλετε να αντικαταστήσετε το υπάρχον αρχείο;")
            if not overwrite:
                messagebox.showwarning("Σφάλμα", 'Δεν ήταν δυνατή η αποθήκευση των προβλέψεων.')
        if not save_path:
            messagebox.showerror("Σφάλμα Αρχείου", "Δεν έχει επιλεγεί αρχείο!")
        df.to_excel(save_path, index=False)
    except Exception as e:
        messagebox.showwarning("Σφάλμα", "Δεν ήταν δυνατή η αποθήκευση των προβλέψεων.\n" + str(e))
        
        
# Συνάρτηση για εμφάνιση μηνύματος σφάλματος αν δεν έχει προηγηθεί η εκπαίδευση του μοντέλου πρόβλεψης
# Θα εξαρτηθεί απο τον κώδικα που θα γράψει ο Ραφαήλ εάν είναι χρήσιμη
def model_check():
    if MODEL is None:
        messagebox.showerror("Σφάλμα", "Πρέπει πρώτα να εκπαιδεύσετε το μοντέλο!")
        
        
# Συναρτήσεις GUI για διαχείριση κλικ κουμπιών
# Λειτουργία κουμπιού για την Εκπαίδευση του μοντέλου πρόβλεψης
def on_train():
    text_area.delete("1.0", tk.END)
    metrics = train_knn()- #placeholder
    messagebox.showinfo("Ενημέρωση", "Η εκπαίδευση του μοντέλου πρόβλεψης ολοκληρώθηκε!")
    text_area.insert(tk.END, "~~~ Εκπαίδευση Μοντέλου ~~~\n\n" + metrics)

# Λειτουργία κουμπιού για την πρόβλεψη απαντήσεων πελατών νέας καμπάνιας 
def on_predict():
    text_area.delete("1.0", tk.END)
    predictions = predict_new_customers() #placeholder
    messagebox.showinfo("Ενημέρωση", "Η πρόβλεψη ανταπόκρισης πελατών νεάς καμπάνιας ολοκληρώθηκε!")
    text_area.insert(tk.END, "~~~ Πρόβλεψη για Νέους Πελάτες ~~~ \n\n" + predictions)   

# Δημιουργία του κύριου παραθύρου GUI
root = tk.Tk()
root.title("Εφαρμογή Προβλέψεων Καμπάνιας")
root.geometry("800x600")

# Πλαίσιο για τα κουμπιά
button_frame = tk.Frame(root)
button_frame.pack(pady=15)

# Κουμπί Εκπαίδευσης Μοντέλου
btn_train = tk.Button(button_frame, text="Εκπαίδευση Μοντέλου",font=('Arial', 12),
bg='#0095fc', fg='white', width=50, command=on_train)
btn_train.grid(row=0, column=0, padx=25)

#Κουμπί Πρόβλεψης Ανταπόκρισης Νέων Πελατών
btn_predict = tk.Button(button_frame, text="Πρόβλεψη Ανταπόκρισης Νέων Πελατών", font=('Arial', 12),
bg='#41b130', fg='white', width=50, command=on_predict)
btn_predict.grid(row=1, column=0, padx=15)

# Περιοχή κειμένου με κύλιση για εμφάνιση αποτελεσμάτων
text_area = scrolledtext.ScrolledText(root, width=80, height=25, font=('Arial', 10))
text_area.pack(padx=15, pady=15)

# Αρχή του βρόχου του GUI
root.mainloop()