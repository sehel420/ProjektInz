import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, BOTH, Frame, OptionMenu, StringVar, Canvas, Frame
from typing import List
dataframes = []
file_names = []
generation_lock = False
## LOAD FILES
def select_files():
    global dataframes, file_names, generation_lock
    file_paths = filedialog.askopenfilenames(filetypes=[["CSV files", "*.csv"]])
    if not file_paths:
        return
    generation_lock = False
    dataframes = [pd.read_csv(path) for path in file_paths]
    file_names = [os.path.basename(path) for path in file_paths]
    messagebox.showinfo("Files Loaded", f"Loaded {len(file_names)} files.")
    
def display_dataset_info(df: pd.DataFrame, file_names):
    info = {
        "File Name": file_names,
        "Number of Rows": len(df),
        "Number of Columns": len(df.columns),
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
    }
    formatted_info = "\n".join(f"{key}: {value}" for key, value in info.items())
    return formatted_info

## LOADED FILE INFO
def show_dataset_info():
    global dataframes, file_names, generation_lock
    if not dataframes:
        messagebox.showerror("Error", "No data loaded. Please load data first.")
        return
    result_window = Toplevel(root)
    result_window.title("Dataset Info")
    Label(result_window, text="Dataset Info", font=("Arial", 14)).pack(pady=15)
    for df, file_name in zip(dataframes, file_names):
        info = display_dataset_info(df, file_name)
        Label(result_window, text=info, justify="center", font=("Arial", 12)).pack(pady=5)
    if not generation_lock:
        Button(result_window, text="Check for inconsistencies", command=lambda: show_inconsistencies(dataframes)).pack(pady=10)
        return
    
## INCONSISTENCIES PART
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

def detect_inconsistencies(df: pd.DataFrame):
    grouped = df.groupby(list(df.columns[:-1]))
    inconsistent_groups = []
    for group, rows in grouped:
        if len(rows[df.columns[-1]].unique()) > 1:
            inconsistent_groups.append(rows)
    return inconsistent_groups

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

## GREDDY HEURISTICS USED FOR CREATING RULES, GENERATION OF RULES PART
def greedy_heuristic_decision_rules(df):
    attributes = list(df.columns[:-1])
    decision_col = df.columns[-1]
    rules = []
    for _, row in df.iterrows():
        Q = set()
        temp_df = df.copy()
        while len(temp_df[decision_col].unique()) > 1:
            for attr in attributes:
                if attr not in Q:
                    temp_df = temp_df[temp_df[attr] == row[attr]]
                    Q.add(attr)
                    break
        rule = {attr: row[attr] for attr in Q}
        rule['decision'] = row[decision_col]
        rules.append(rule)
    return rules

def generate_decision_rules():
    global dataframes, file_names, generation_lock
    if generation_lock:
        messagebox.showerror("Error", "Rule generation is locked. Reload files to enable rule generation.")
        return
    if not dataframes:
        messagebox.showerror("Error", "No data loaded. Please load data first.")
        return
    results = []
    for df, file_name in zip(dataframes, file_names):
        rules = greedy_heuristic_decision_rules(df)
        unique_rules = {frozenset(rule.items()): rule for rule in rules}
        rule_lengths = [len(rule) - 1 for rule in rules]
        rule_supports = [1 / len(rules) for _ in rules]
        analysis = {
            "file_name": file_name,
            "total_rules_count": len(rules),
            "unique_rules_count": len(unique_rules),
            "min_length": min(rule_lengths),
            "avg_length": np.mean(rule_lengths),
            "max_length": max(rule_lengths),
            "min_support": min(rule_supports),
            "avg_support": np.mean(rule_supports),
            "max_support": max(rule_supports),
            "unique_rules": list(unique_rules.values()),
            "all_rules": rules
        }
        results.append(analysis)
    display_generated_rules_results(results)

def display_generated_rules_results(results: List[dict]):
    stats_window = Toplevel(root)
    stats_window.title("Generated Rules Statistics")
    stats_window.geometry("400x600")

    canvas = Canvas(stats_window)
    scrollbar = Scrollbar(stats_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((200, 0), window=scrollable_frame, anchor="n")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for analysis in results:
        container = Frame(scrollable_frame)
        container.pack(fill="x", pady=10)

        Label(container, text=f"File: {analysis['file_name']}", font=("Arial", 14)).pack(pady=5, anchor="center")
        
        stats_text = (
            f"Total Rules: {analysis['total_rules_count']}\n"
            f"Unique Rules: {analysis['unique_rules_count']}\n"
            f"Min Length: {analysis['min_length']}\n"
            f"Avg Length: {analysis['avg_length']:.2f}\n"
            f"Max Length: {analysis['max_length']}\n"
            f"Min Support: {analysis['min_support']:.4f}\n"
            f"Avg Support: {analysis['avg_support']:.4f}\n"
            f"Max Support: {analysis['max_support']:.4f}\n"
        )
        Label(container, text=stats_text, justify="center", font=("Arial", 12)).pack(pady=5)

        Button(container, text="Show Unique Rules", command=lambda a=analysis: show_rules(a["unique_rules"], a["file_name"], "Unique Rules")).pack(pady=5)
        Button(container, text="Show All Rules", command=lambda a=analysis: show_rules(a["all_rules"], a["file_name"], "All Rules")).pack(pady=5)
        
def show_rules(rules, file_name, title):
    rules_window = Toplevel(root)
    rules_window.title(f"{title} - {file_name}")
    text_widget = Text(rules_window, wrap="word", font=("Arial", 12))
    scrollbar = Scrollbar(rules_window, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    for index, rule in enumerate(rules, start=1):
        
        text_widget.insert("end", f"{index}. Condition: {rule}\nDecision: {rule['decision']}\n\n")
    
    Button(rules_window, text="Save to CSV", command=lambda: save_rules_to_csv(rules, file_name)).pack(pady=10)
    
## SAVING GENERATED RULES TO CSV FILE, YOU NEED TO SAVE IT TO OPTYMALIZATE
def save_rules_to_csv(rules, file_name):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    formatted_rules = []
    for index, rule in enumerate(rules, start=1):
        decision = rule.pop('decision')
        length = len(rule)
        support = 1 / len(rules)
        rule_text = ", ".join([f"{k}={v}" for k, v in rule.items()])
        formatted_rules.append({"Index": index, "Rule": rule_text, "Decision": decision, "Support": support, "Length": length})

    df = pd.DataFrame(formatted_rules)
    df.to_csv(file_path, index=False)
    messagebox.showinfo("Success", "Rules saved successfully.")
    
## OPTYMALIZATION
def select_files_optymalization():
    global dataframes, file_names, generation_lock
    messagebox.showinfo("Info", "Load atleast 2 files with generated decision rules.")
    file_paths = filedialog.askopenfilenames(filetypes=[["CSV files", "*.csv"]])
    if not file_paths:
        return
    dataframes = [pd.read_csv(path) for path in file_paths]
    file_names = [os.path.basename(path) for path in file_paths]
    messagebox.showinfo("Files Loaded", f"Loaded {len(file_names)} files.")
    generation_lock = True
    optimize_rules()
    
def optimize_rules():
    global dataframes, file_names
    if len(dataframes) < 2:
        messagebox.showerror("Error", "Optimization requires at least two CSV files.")
        return
    
    for df in dataframes:
        if list(df.columns) != ["Index", "Rule", "Decision", "Support", "Length"]:
            messagebox.showerror("Error", "CSV file structure is incorrect. Expected columns: Index, Rule, Decision, Support, Length")
            return
    
    optimization_window = Toplevel(root)
    optimization_window.title("Optimize Rules")
    Label(optimization_window, text="Select Optimization Method:", font=("Arial", 12)).pack(pady=5)
    
    optimization_method = StringVar(optimization_window)
    optimization_method.set("Support")
    methods = ["Support", "Length"]
    OptionMenu(optimization_window, optimization_method, *methods).pack(pady=5)
    
    Button(optimization_window, text="Run Optimization", command=lambda: run_optimization(optimization_method.get(), optimization_window)).pack(pady=10)

def run_optimization(method, parent_window):
    global dataframes, file_names
    combined_df = pd.concat(dataframes)
    optimized_rules = {}
    
    for _, row in combined_df.iterrows():
        index = row["Index"]
        rule = f"{row['Rule']}, decision={row['Decision']}"
        value = row["Support"] if method == "Support" else row["Length"]
        
        if index not in optimized_rules:
            optimized_rules[index] = (rule, value)
        else:
            existing_rule, existing_value = optimized_rules[index]
            if value > existing_value if method == "Support" else value < existing_value:
                optimized_rules[index] = (rule, value)
            elif value == existing_value:
                optimized_rules[index] = (f"{existing_rule};{rule}", value)
    
    display_optimized_rules(optimized_rules, parent_window)

def display_optimized_rules(optimized_rules, parent_window):
    result_window = Toplevel(parent_window)
    result_window.title("Optimized Rules")
    
    text_widget = Text(result_window, wrap="word", font=("Arial", 12))
    scrollbar = Scrollbar(result_window, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    for index, (rule, _) in optimized_rules.items():
        text_widget.insert("end", f"{index}: {rule}\n\n")
    
    Button(result_window, text="Save to CSV", command=lambda: save_optimized_rules_to_csv(optimized_rules)).pack(pady=10)

def save_optimized_rules_to_csv(optimized_rules):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    
    df = pd.DataFrame([(index, rule) for index, (rule, _) in optimized_rules.items()], columns=["Index", "Rule"])
    df.to_csv(file_path, index=False)
    messagebox.showinfo("Success", "Optimized rules saved successfully.")
    
    
def main():
    global root
    root = Tk()
    root.title("CSV Rule Analysis")
    root.geometry("400x600")
    Label(root, text="CSV Rule Analysis Tool", font=("Arial", 16)).pack(pady=20)
    Button(root, text="Select CSV Files", command=select_files, font=("Arial", 12)).pack(pady=30)
    Button(root, text="Dataset Info", command=show_dataset_info, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Generate decision rules", command=generate_decision_rules, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Optimize Rules", command=select_files_optymalization, font=("Arial", 12)).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()
