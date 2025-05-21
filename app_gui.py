# app_gui.py

import tkinter as tk
from tkinter import messagebox
from spell_model import SpellChecker


# Load dictionary
# dictionary = load_dictionary()

# Spell checker logic
def check_spelling():
    input_word = entry.get().strip().lower()
    if not input_word:
        messagebox.showwarning("Input Error", "Please enter a word.")
        return

    # suggestion = get_closest_word(input_word, dictionary)
    spc = SpellChecker()
    suggestion = spc.suggest(word=input_word)
    result_label.config(text=f"Suggestion: {suggestion}")

# GUI setup
root = tk.Tk()
root.title("Smart Spell Checker")
root.geometry("800x600")
tk.Label(root, text="Enter a word:").pack(pady=5)
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

check_button = tk.Button(root, text="Check Spelling", command=check_spelling)
check_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), fg="white")
result_label.pack(pady=10)

root.mainloop()
