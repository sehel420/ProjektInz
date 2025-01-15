import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y
from typing import List

# Step 1: Loading and verifying data
def load_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """Load multiple CSV files into a list of DataFrames."""
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading {path}: {e}")
    return dataframes

def verify_data(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwanie niespójności"""
    grouped = df.groupby(list(df.columns[:-1]))
    resolved_rows = []

    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            # Handle inconsistent rows
            common_decision = rows[df.columns[-1]].mode()[0]  # Most common decision
            new_row = list(group) + [common_decision]
            resolved_rows.append(new_row)
        else:
            resolved_rows.append(list(group) + [rows[df.columns[-1]].iloc[0]])

    resolved_df = pd.DataFrame(resolved_rows, columns=df.columns)
    return resolved_df

# Indukcja Reguł Decyzyjnych, wykorzystanie Heurystyki RM
def induce_rules(df: pd.DataFrame) -> List[dict]:
    rules = []
    for _, row in df.iterrows(): 
        condition = {col: int(row[col]) if isinstance(row[col], (np.integer, np.int64)) else row[col] for col in df.columns[:-1] if pd.notna(row[col])}
        decision = row[df.columns[-1]]
        rules.append({"condition": condition, "decision": decision})
    return rules

# Step 3: Calculate rule lengths and optimize
def calculate_rule_length(rule: dict) -> int:
    return len(rule["condition"])

def optimize_rules(rules: List[dict]) -> List[dict]:
    optimized_rules = []
    for rule in rules:
        length = calculate_rule_length(rule)
        optimized_rules.append({**rule, "length": length})
    return sorted(optimized_rules, key=lambda r: r["length"])

# Step 4: Analyze results
def analyze_rules(rules: List[dict]) -> dict:
    unique_rules = {frozenset(rule["condition"].items()): rule for rule in rules}
    lengths = [rule["length"] for rule in unique_rules.values()]

    return {
        "unique_rules_count": len(unique_rules),
        "min_length": min(lengths),
        "avg_length": np.mean(lengths),
        "max_length": max(lengths),
        "unique_rules": list(unique_rules.values()),
    }

def plot_analysis(analysis: dict):
    fig, ax = plt.subplots()
    ax.bar(['Min Length', 'Avg Length', 'Max Length'], [
        analysis['min_length'], 
        analysis['avg_length'], 
        analysis['max_length']
    ])
    ax.set_title('Rule Length Analysis')
    ax.set_ylabel('Length')
    plt.show()

# GUI
def main():
    def select_files():
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not file_paths:
            return

        dataframes = load_csv_files(file_paths)
        all_rules = []

        for df in dataframes:
            verified_df = verify_data(df)
            rules = induce_rules(verified_df)
            optimized_rules = optimize_rules(rules)
            all_rules.extend(optimized_rules)

        analysis = analyze_rules(all_rules)
        display_results(analysis)

    def display_results(analysis):
        result_window = Toplevel(root)
        result_window.title("Analysis Results")

        Label(result_window, text="Verification and Rule Analysis", font=("Arial", 14)).pack(pady=10)

        stats_text = f"""
Unique Rules Count: {analysis['unique_rules_count']}
Min Length: {analysis['min_length']}
Avg Length: {analysis['avg_length']:.2f}
Max Length: {analysis['max_length']}
"""
        Label(result_window, text=stats_text, justify="left", font=("Arial", 12)).pack(pady=5)

        scrollbar = Scrollbar(result_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_box = Text(result_window, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        for rule in analysis['unique_rules']:
            text_box.insert("end", f"Condition: {rule['condition']}, Decision: {rule['decision']}, Length: {rule['length']}\n")
        text_box.pack(expand=True, fill="both", padx=10, pady=10)
        scrollbar.config(command=text_box.yview)

        Button(result_window, text="Show Plot", command=lambda: plot_analysis(analysis)).pack(pady=10)

    root = Tk()
    root.title("CSV Rule Analysis")
    root.geometry("600x300")

    Label(root, text="Optymalizacja Reguł decyzyjnych", font=("Arial", 16)).pack(pady=20)

    Button(root, text="Wybierz Pliki", command=select_files, font=("Arial", 12)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
