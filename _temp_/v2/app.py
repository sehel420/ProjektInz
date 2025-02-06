import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, StringVar
from typing import List

dataframes = []
file_names = []

# Wczytanie danych
def load_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading {path}: {e}")
    return dataframes

# Informacje o wczytanych plikach
def display_dataset_info(df: pd.DataFrame, file_name):
    info = {
        "File Name": file_name,
        "Number of Rows": len(df),
        "Number of Columns": len(df.columns),
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
    }
    formatted_info = "\n".join(f"{key}: {value}" for key, value in info.items())
    return formatted_info

# Sprawdzanie niespójności
def detect_inconsistencies(df: pd.DataFrame):
    # Grupujemy po wszystkich kolumnach poza ostatnią (decyzyjną)
    grouped = df.groupby(list(df.columns[:-1]))
    inconsistent_groups = []
    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            inconsistent_groups.append(rows)
    return inconsistent_groups

# Usuwanie niespójności
def remove_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(list(df.columns[:-1]))
    resolved_rows = []
    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            common_decision = rows[df.columns[-1]].mode()[0]
            new_row = list(group) + [common_decision]
            resolved_rows.append(new_row)
        else:
            resolved_rows.append(list(group) + [rows[df.columns[-1]].iloc[0]])
    resolved_df = pd.DataFrame(resolved_rows, columns=df.columns)
    return resolved_df

# Indukcja reguł decyzyjnych wykorzystując heurystykę Row Matching
def induce_rules(df: pd.DataFrame) -> List[dict]:
    """
    Generowanie reguł decyzyjnych na podstawie każdej obserwacji (wiersza)
    przyjmując, że kolumny poza ostatnią tworzą warunek,
    a ostatnia kolumna – decyzję.
    """
    rules = []
    for _, row in df.iterrows():
        condition = {}
        for col in df.columns[:-1]:
            value = row[col]
            if pd.notna(value):  # Ignorujemy brakujące wartości
                # Konwertujemy wartość do typu float (lub int, jeśli wolisz)
                if isinstance(value, (np.float64, np.float32)):
                    value = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    value = int(value)
                condition[col] = value
        # Konwersja decyzji do odpowiedniego typu
        decision = row[df.columns[-1]]
        if isinstance(decision, (np.float64, np.float32)):
            decision = float(decision)
        elif isinstance(decision, (np.int64, np.int32)):
            decision = int(decision)
        rules.append({"condition": condition, "decision": decision})
    return rules

# Obliczenie długości reguły (liczba warunków)
def calculate_rule_length(rule: dict) -> int:
    return len(rule["condition"])

# Obliczenie wsparcia reguły
def calculate_rule_support(rule: dict, df: pd.DataFrame) -> float:
    """
    Oblicza wsparcie reguły jako liczbę pasujących wierszy podzieloną przez całkowitą liczbę wierszy.
    """
    condition = rule["condition"]
    filtered_df = df
    for col, value in condition.items():
        filtered_df = filtered_df[filtered_df[col] == value]
    support = len(filtered_df) / len(df)
    return round(support, 4)

# Funkcja wyświetlająca wykres analizy
def plot_analysis(analysis: dict, criterion: str):
    fig, ax = plt.subplots()
    ax.bar([f"Min {criterion.title()}", f"Avg {criterion.title()}", f"Max {criterion.title()}"],
           [analysis[f"min_{criterion}"], analysis[f"avg_{criterion}"], analysis[f"max_{criterion}"]])
    ax.set_title(f'Rule {criterion.title()} Analysis')
    ax.set_ylabel(criterion.title())
    plt.show()

# GUI
def main():
    
    def select_files():
        global dataframes
        global file_names
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not file_paths:
            return
        dataframes = load_csv_files(file_paths)
        file_names = [os.path.basename(path) for path in file_paths]
        messagebox.showinfo("Files Loaded", f"You have loaded {len(file_names)} files.")
        show_dataset_info()
    
    def show_dataset_info():
        global dataframes
        global file_names
        if not dataframes:
            messagebox.showerror("Error", "No data loaded. Please load data first.")
            return
        result_window = Toplevel(root)
        result_window.title("Dataset Info")
        Label(result_window, text="Dataset Info", font=("Arial", 14)).pack(pady=15)
        for df, file_name in zip(dataframes, file_names):
            info = display_dataset_info(df, file_name)
            Label(result_window, text=info, justify="center", font=("Arial", 12)).pack(pady=5)
        Button(result_window, text="Check for inconsistencies", command=lambda: show_inconsistencies(dataframes)).pack(pady=10)
    
    def show_inconsistencies(dataframes):
        for df in dataframes:
            inconsistencies = detect_inconsistencies(df)
            if inconsistencies:
                messagebox.showwarning("Inconsistencies Found", f"Number of inconsistent groups: {len(inconsistencies)}")
                response = messagebox.askyesno("Inconsistencies Found", "Do you want to remove inconsistencies?")
                if response:
                    remove_all_inconsistencies(dataframes)
                else:
                    messagebox.showinfo("Action Skipped", "Inconsistencies were not removed.")
                return
        messagebox.showinfo("Info", "No inconsistencies were found")
    
    def remove_all_inconsistencies(dataframes):
        for i, df in enumerate(dataframes):
            dataframes[i] = remove_inconsistencies(df)
        messagebox.showinfo("Inconsistencies Removed", "All inconsistencies have been resolved.")
    
    def generate_decision_rules():
        """
        Generowanie reguł decyzyjnych za pomocą heurystyki Row Matching,
        obliczenie statystyk (wszystkie reguły, unikalne reguły, długość, wsparcie)
        oraz wyświetlenie wyników w oknie GUI.
        """
        global dataframes
        if not dataframes:
            messagebox.showerror("Error", "No data loaded. Please load data first.")
            return
        
        # Sprawdzenie niespójności w każdej z wczytanych ram danych
        for df in dataframes:
            if detect_inconsistencies(df):
                messagebox.showwarning("Info", "Inconsistencies were found in one of the datasets!")
                return
        
        # Łączymy wszystkie ramki danych w jedną (przyjmujemy, że struktura plików jest taka sama)
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Generujemy reguły – każda obserwacja daje jedną regułę (heurystyka Row Matching)
        all_rules = induce_rules(combined_df)
        
        # Dla każdej reguły obliczamy wsparcie (na podstawie zbioru combined_df)
        for rule in all_rules:
            rule["support"] = calculate_rule_support(rule, combined_df)
        
        total_rules_count = len(all_rules)
        
        # Obliczamy liczbę unikalnych reguł (na podstawie unikalnych warunków)
        unique_rules = {}
        for rule in all_rules:
            key = frozenset(rule["condition"].items())
            # Zakładamy, że dla danej reguły wsparcie i długość będą identyczne
            if key not in unique_rules:
                unique_rules[key] = rule.copy()
        unique_rules_count = len(unique_rules)
        
        # Obliczamy statystyki długości reguł
        rule_lengths = [calculate_rule_length(rule) for rule in all_rules]
        min_length = min(rule_lengths) if rule_lengths else 0
        max_length = max(rule_lengths) if rule_lengths else 0
        avg_length = np.mean(rule_lengths) if rule_lengths else 0
        
        # Obliczamy statystyki wsparcia reguł
        rule_supports = [rule["support"] for rule in all_rules]
        min_support = min(rule_supports) if rule_supports else 0
        max_support = max(rule_supports) if rule_supports else 0
        avg_support = np.mean(rule_supports) if rule_supports else 0
        
        # Zestawienie wyników do wyświetlenia
        analysis = {
            "total_rules_count": total_rules_count,
            "unique_rules_count": unique_rules_count,
            "min_length": min_length,
            "avg_length": avg_length,
            "max_length": max_length,
            "min_support": min_support,
            "avg_support": avg_support,
            "max_support": max_support,
            "unique_rules": list(unique_rules.values())
        }
        
        display_generated_rules_results(analysis)
    
    def display_generated_rules_results(analysis: dict):
        """
        Wyświetlanie wyników analizy wygenerowanych reguł:
         - Liczba wszystkich reguł
         - Liczba unikalnych reguł
         - Minimalna, średnia, maksymalna długość reguł
         - Minimalne, średnie, maksymalne wsparcie reguł
        """
        result_window = Toplevel(root)
        result_window.title("Generated Rules Analysis")
        Label(result_window, text="Generated Rules Analysis", font=("Arial", 14)).pack(pady=10)
        
        stats_text = (
            f"Liczba wszystkich reguł: {analysis['total_rules_count']}\n"
            f"Liczba unikalnych reguł: {analysis['unique_rules_count']}\n\n"
            f"Długość reguł:\n"
            f"  Min: {analysis['min_length']}\n"
            f"  Średnia: {analysis['avg_length']:.2f}\n"
            f"  Max: {analysis['max_length']}\n\n"
            f"Wsparcie reguł:\n"
            f"  Min: {analysis['min_support']:.4f}\n"
            f"  Średnie: {analysis['avg_support']:.4f}\n"
            f"  Max: {analysis['max_support']:.4f}\n"
        )
        Label(result_window, text=stats_text, justify="left", font=("Arial", 12)).pack(pady=5)
        
        scrollbar = Scrollbar(result_window)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        text_box = Text(result_window, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        # Wyświetlamy listę unikalnych reguł
        for rule in analysis['unique_rules']:
            text_box.insert("end", f"Condition: {rule['condition']}, Decision: {rule['decision']}, "
                                     f"Length: {calculate_rule_length(rule)}, Support: {rule['support']}\n")
        text_box.pack(expand=True, fill="both", padx=10, pady=10)
        scrollbar.config(command=text_box.yview)
        
        Button(result_window, text="Show Length Plot", command=lambda: plot_analysis({
            "min_length": analysis["min_length"],
            "avg_length": analysis["avg_length"],
            "max_length": analysis["max_length"]
        }, "length")).pack(pady=10)
        Button(result_window, text="Show Support Plot", command=lambda: plot_analysis({
            "min_support": analysis["min_support"],
            "avg_support": analysis["avg_support"],
            "max_support": analysis["max_support"]
        }, "support")).pack(pady=10)
    
    root = Tk()
    root.title("CSV Rule Analysis")
    root.geometry("400x600")
    
    Label(root, text="CSV Rule Analysis Tool", font=("Arial", 16)).pack(pady=20)
    Button(root, text="Select CSV Files", command=select_files, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Generate decision rules", command=generate_decision_rules, font=("Arial", 12)).pack(pady=30)
    Button(root, text="Dataset Info", command=show_dataset_info, font=("Arial", 12)).pack(pady=30)
    
    root.mainloop()

if __name__ == "__main__":
    main()
