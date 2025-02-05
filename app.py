import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, StringVar, OptionMenu
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
# Informacje O Wczytanych plikach
def display_dataset_info(df: pd.DataFrame,file_names):
    info = {
        "File Name": file_names,
        "Number of Rows": len(df),
        "Number of Columns": len(df.columns),
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
    }
    formatted_info = "\n".join(f"{key}: {value}" for key, value in info.items())
    return formatted_info
def display_root_file_name(file_names):
    info = {
       "File Name" : file_names
    }
    formatted_info = "\n".join(f"{key}: {value}" for key, value in info.items())
    return formatted_info

# Sprawdzanie Niespójności
def detect_inconsistencies(df: pd.DataFrame):
    grouped = df.groupby(list(df.columns[:-1]))
    inconsistent_groups = []

    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            inconsistent_groups.append(rows)

    return inconsistent_groups
# Usuwanie Niespójności
def remove_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(list(df.columns[:-1]))
    resolved_rows = []
    """Usuwanie niespójności"""
    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            common_decision = rows[df.columns[-1]].mode()[0]
            new_row = list(group) + [common_decision]
            resolved_rows.append(new_row)
        else:
            resolved_rows.append(list(group) + [rows[df.columns[-1]].iloc[0]])

    resolved_df = pd.DataFrame(resolved_rows, columns=df.columns)
    return resolved_df

# Indukcja Reguł Decyzyjnych, wykorzystanie Heurystyki RM
def induce_rules(df: pd.DataFrame) -> List[dict]:
    """Generowanie reguł decyzyjnych na podstawie danych."""
    rules = []
    for _, row in df.iterrows():
        condition = {}
        for col in df.columns[:-1]:
            value = row[col]
            if pd.notna(value):  # Ignorowanie brakujących wartości
                condition[col] = int(value) if isinstance(value, (np.integer, np.int64)) else value
        decision = int(row[df.columns[-1]]) if isinstance(row[df.columns[-1]], (np.integer, np.int64)) else row[df.columns[-1]]
        rules.append({"condition": condition, "decision": decision})
    return rules
# Step 3: Calculate rule lengths and optimize
def calculate_rule_length(rule: dict) -> int:
    return len(rule["condition"])

def calculate_rule_support(rule: dict, df: pd.DataFrame) -> float:
    """Oblicza wsparcie reguły jako liczbę pasujących wierszy podzieloną przez całkowitą liczbę wierszy."""
    condition = rule["condition"]
    filtered_df = df
    for col, value in condition.items():
        filtered_df = filtered_df[filtered_df[col] == value]
    
    # Wsparcie to stosunek pasujących wierszy do wszystkich wierszy
    support = len(filtered_df) / len(df)
    return round(support, 4) 

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

def analyze_rules(rules: List[dict], metric: str, all_rules: List[dict] = None) -> dict:
    """Analiza unikalności reguł i metryk (długość, wsparcie)."""
    if not rules:
        return {
            "unique_rules_count": 0,
            "min_metric": 0,
            "avg_metric": 0,
            "max_metric": 0,
            "unique_rules": [],
        }

    unique_rules = {}
    for rule in rules:
        key = frozenset(rule["condition"].items())
        if key not in unique_rules:
            unique_rules[key] = rule.copy()
            unique_rules[key]["support_metric"] = 0  # Inicjalizacja wsparcia
        unique_rules[key]["support_metric"] += 1  # Zliczanie wystąpień reguły

    # Obliczenie wybranej metryki (długość lub wsparcie)
    for rule in unique_rules.values():
        if metric == "length":
            rule["length_metric"] = len(rule["condition"])
    
    metrics = [rule.get(f"{metric}_metric", 0) for rule in unique_rules.values()]

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
       
       # show_inconsistencies(dataframes)
        Button(result_window, text="Check for inconsistencies", command=lambda: show_inconsistencies(dataframes)).pack(pady=10)
            
    def show_inconsistencies(dataframes):
        for df in dataframes:
            inconsistencies = detect_inconsistencies(df)
        if inconsistencies:
                messagebox.showwarning("Inconsistencies Found", f"Number of inconsistent groups: {len(inconsistencies)}")
                response=messagebox.askyesno("Inconsistencies Found","Do you want to remove inconsistencies?")
                if response:
                         remove_all_inconsistencies(dataframes)
                else:
                         messagebox.showinfo("Action Skipped", "Inconsistencies were not removed.")
        else:
                messagebox.showwarning("Info","No inconsistencies were found")
                
    def remove_all_inconsistencies(dataframes):
        for i, df in enumerate(dataframes):
            dataframes[i] = remove_inconsistencies(df)
        messagebox.showinfo("Inconsistencies Removed", "All inconsistencies have been resolved.")

    def generate_decision_rules():
        """Generowanie reguł decyzyjnych z analityką i wyświetlanie wyników w GUI."""
        global dataframes
        if not dataframes:
            messagebox.showerror("Error", "No data loaded. Please load data first.")
            return

        all_rules = []
        for df in dataframes:
            inconsistencies = detect_inconsistencies(df)
            if not inconsistencies:
                rules = induce_rules(df)
                all_rules.extend(rules)
            else:
                messagebox.showwarning("Info", "Inconsistencies were found!")
                return
        
        length_analysis = analyze_rules(all_rules, metric="length")
        support_analysis = analyze_rules(all_rules, metric="support", all_rules=all_rules)
        
        display_generated_rules_results(length_analysis, support_analysis)

    def display_generated_rules_results(length_analysis: dict, support_analysis: dict):
        """Wyświetlanie wyników analizy reguł dla długości i wsparcia."""
        result_window = Toplevel(root)
        result_window.title("Generated Rules Analysis")

        Label(result_window, text="Generated Rules Analysis", font=("Arial", 14)).pack(pady=10)

        stats_text = f"""
        Unique Rules Count: {length_analysis['unique_rules_count']}
        Min Length: {length_analysis['min_metric']:.2f}
        Avg Length: {length_analysis['avg_metric']:.2f}
        Max Length: {length_analysis['max_metric']:.2f}

        Min Support: {support_analysis['min_metric']:.2f}
        Avg Support: {support_analysis['avg_metric']:.2f}
        Max Support: {support_analysis['max_metric']:.2f}
        """
        Label(result_window, text=stats_text, justify="left", font=("Arial", 12)).pack(pady=5)

        scrollbar = Scrollbar(result_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_box = Text(result_window, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        for rule in length_analysis['unique_rules']:
            text_box.insert("end", f"Condition: {rule['condition']}, Decision: {rule['decision']}, Length: {rule.get('length_metric', 0)}, Support: {rule.get('support_metric', 0)}\n")
        text_box.pack(expand=True, fill="both", padx=10, pady=10)
        scrollbar.config(command=text_box.yview)

        Button(result_window, text="Show Length Plot", command=lambda: plot_analysis(length_analysis, "length")).pack(pady=10)
        Button(result_window, text="Show Support Plot", command=lambda: plot_analysis(support_analysis, "support")).pack(pady=10)

    
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
    root.geometry("400x600")

    Label(root, text="CSV Rule Analysis Tool", font=("Arial", 16)).pack(pady=20)
    
    
            
    Button(root, text="Select CSV Files", command=select_files, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Generate decision rules", command=generate_decision_rules, font=("Arial", 12)).pack(pady=30)
    Button(root, text="Dataset Info", command=show_dataset_info, font=("Arial", 12)).pack(pady=30)
  

    root.mainloop()

if __name__ == "__main__":
    main()