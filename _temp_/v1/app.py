import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, StringVar, OptionMenu
from typing import List

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
# Weryfikacja danych
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

def calculate_rule_support(rule: dict, df: pd.DataFrame) -> int:
    """Calculate the support of a rule based on its condition."""
    condition = rule["condition"]
    filtered_df = df
    for col, value in condition.items():
        filtered_df = filtered_df[filtered_df[col] == value]
    return len(filtered_df)

def optimize_rules(rules: List[dict], df: pd.DataFrame, criterion: str) -> List[dict]:
    """Optimize rules based on the specified criterion (length or support)."""
    optimized_rules = []
    for rule in rules:
        if criterion == "length":
            metric = calculate_rule_length(rule)
        elif criterion == "support":
            metric = calculate_rule_support(rule, df)
        else:
            raise ValueError("Invalid optimization criterion")
        optimized_rules.append({**rule, "metric": metric})
    return sorted(optimized_rules, key=lambda r: r["metric"] if criterion == "length" else -r["metric"])

# Step 4: Analyze results
def analyze_rules(rules: List[dict]) -> dict:
    unique_rules = {frozenset(rule["condition"].items()): rule for rule in rules}
    metrics = [rule["metric"] for rule in unique_rules.values()]

    return {
        "unique_rules_count": len(unique_rules),
        "min_metric": min(metrics),
        "avg_metric": np.mean(metrics),
        "max_metric": max(metrics),
        "unique_rules": list(unique_rules.values()),
    }

def plot_analysis(analysis: dict, criterion: str):
    fig, ax = plt.subplots()
    ax.bar([f"Min {criterion.title()}", f"Avg {criterion.title()}", f"Max {criterion.title()}"], [
        analysis['min_metric'], 
        analysis['avg_metric'], 
        analysis['max_metric']
    ])
    ax.set_title(f'Rule {criterion.title()} Analysis')
    ax.set_ylabel(criterion.title())
    plt.show()

# GUI
def main():
    def select_files():
        file_paths = filedialog.askopenfilenames(filetypes=[["CSV files", "*.csv"]])
        if not file_paths:
            return

        dataframes = load_csv_files(file_paths)
        all_rules = []
        file_names = [os.path.basename(path) for path in file_paths]  # Pobierz nazwy plików

        for df in dataframes:
            verified_df = verify_data(df)
            rules = induce_rules(verified_df)
            all_rules.append((rules, verified_df))

        show_optimization_options(all_rules, file_names)

    def show_optimization_options(rules_and_dfs, file_names):
        option_window = Toplevel(root)
        option_window.title("Optimization Options")

        Label(option_window, text="Select Optimization Criterion", font=("Arial", 14)).pack(pady=10)

        # Display loaded files
        loaded_files_text = f"Wczytane pliki:\n" + "\n".join(file_names)
        Label(option_window, text=loaded_files_text, font=("Arial", 10), justify="left").pack(pady=10)

        criterion = StringVar(value="length")

        Button(option_window, text="Optimize by Length", command=lambda: optimize_and_display(rules_and_dfs, "length")).pack(pady=10)
        Button(option_window, text="Optimize by Support", command=lambda: optimize_and_display(rules_and_dfs, "support")).pack(pady=10)

    def optimize_and_display(rules_and_dfs, criterion):
        all_optimized_rules = []
        for rules, df in rules_and_dfs:
            optimized_rules = optimize_rules(rules, df, criterion)
            all_optimized_rules.extend(optimized_rules)
        
        analysis = analyze_rules(all_optimized_rules)
        display_results(analysis, criterion)

    def display_results(analysis, criterion):
        result_window = Toplevel(root)
        result_window.title("Analysis Results")

        Label(result_window, text="Verification and Rule Analysis", font=("Arial", 14)).pack(pady=10)

        stats_text = f"""
Unique Rules Count: {analysis['unique_rules_count']}
Min {criterion.title()}: {analysis['min_metric']}
Avg {criterion.title()}: {analysis['avg_metric']:.2f}
Max {criterion.title()}: {analysis['max_metric']}
"""
        Label(result_window, text=stats_text, justify="left", font=("Arial", 12)).pack(pady=5)

        scrollbar = Scrollbar(result_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_box = Text(result_window, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        for rule in analysis['unique_rules']:
            text_box.insert("end", f"Condition: {rule['condition']}, Decision: {rule['decision']}, Metric: {rule['metric']}\n")
        text_box.pack(expand=True, fill="both", padx=10, pady=10)
        scrollbar.config(command=text_box.yview)

        Button(result_window, text="Show Plot", command=lambda: plot_analysis(analysis, criterion)).pack(pady=10)

    root = Tk()
    root.title("CSV Rule Analysis")
    root.geometry("400x200")

    Label(root, text="CSV Rule Analysis Tool", font=("Arial", 16)).pack(pady=20)

    Button(root, text="Select CSV Files", command=select_files, font=("Arial", 12)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
