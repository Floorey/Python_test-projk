import tkinter as tk
from tkinter import messagebox
import datetime


to_do_list = [
    {"Aufgabe": "Krankenkasse", "Termin": "Montag"},
    {"Aufgabe": "Rentenkasse", "Termin": "Montag"},
    {"Aufgabe": "Versicherungen", "Termin": "Clarck App (Berufsunfähigkerit, Rechtsschutz)"},
    {"Aufgabe": "Terminvereinbarung Tagesklinik Esslingen ADHS", "Termin": "über das Chefarztsekretariat Frau Bosse, Telefon: 0711 3103-3201"}
]


def anzeigen_aufgaben_fuer_heute():
    heute = datetime.datetime.now().strftime("%A")
    aufgaben_heute = [task["Aufgabe"] for task in to_do_list if task["Termin"] == heute]
    if aufgaben_heute:
        return "\n".join(aufgaben_heute)
    else:
        return "Keine geplanten Aufgaben für heute."


def anzeigen_to_do_liste():
    to_do_text = "\n".join([f"{i+1}. {task['Aufgabe']}: {task['Termin']}" for i, task in enumerate(to_do_list)])
    return to_do_text


def neue_aufgabe_hinzufuegen():
    aufgabe = aufgabe_entry.get()
    termin = termin_entry.get()
    if aufgabe and termin:
        to_do_list.append({"Aufgabe": aufgabe, "Termin": termin})
        aktualisiere_to_do_liste()
        aufgabe_entry.delete(0, tk.END)
        termin_entry.delete(0, tk.END)
        messagebox.showinfo("Erfolg", "Die Aufgabe wurde erfolgreich hinzugefügt.")
    else:
        messagebox.showerror("Fehler", "Bitte gib sowohl die Aufgabe als auch den Termin ein.")

def aktualisiere_to_do_liste():
    aufgaben_text = anzeigen_aufgaben_fuer_heute()
    aufgaben_label.config(text=aufgaben_text)
    to_do_liste_text = anzeigen_to_do_liste()
    to_do_liste_label.config(text=to_do_liste_text)

root = tk.Tk()
root.title("To-Do-Liste")


aufgaben_label = tk.Label(root, text=anzeigen_aufgaben_fuer_heute(), justify=tk.LEFT, padx=10, pady=10)
aufgaben_label.pack(side=tk.LEFT)


to_do_liste_label = tk.Label(root, text=anzeigen_to_do_liste(), justify=tk.LEFT, padx=10, pady=10)
to_do_liste_label.pack(side=tk.RIGHT)


neue_aufgabe_frame = tk.Frame(root)
neue_aufgabe_frame.pack(pady=10)

tk.Label(neue_aufgabe_frame, text="Aufgabe:").grid(row=0, column=0, padx=5, pady=5)
aufgabe_entry = tk.Entry(neue_aufgabe_frame)
aufgabe_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(neue_aufgabe_frame, text="Termin:").grid(row=1, column=0, padx=5, pady=5)
termin_entry = tk.Entry(neue_aufgabe_frame)
termin_entry.grid(row=1, column=1, padx=5, pady=5)

hinzufuegen_button = tk.Button(neue_aufgabe_frame, text="Hinzufügen", command=neue_aufgabe_hinzufuegen)
hinzufuegen_button.grid(row=2, columnspan=2, padx=5, pady=5)


root.mainloop()
