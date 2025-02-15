import os
import base64
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, BOTH, Frame, OptionMenu, StringVar, Canvas, Frame, PhotoImage
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
    result_window.geometry("700x600")

    container = Frame(result_window)
    container.pack(fill="both", expand=True)

    canvas = Canvas(container)

    v_scrollbar = Scrollbar(container, orient="vertical", command=canvas.yview)
    h_scrollbar = Scrollbar(result_window, orient="horizontal", command=canvas.xview)

    scrollable_frame = Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    v_scrollbar.pack(side="right", fill="y")
    h_scrollbar.pack(side="bottom", fill="x")
    Label(scrollable_frame, text="Dataset Info", font=("Arial", 14)).pack(pady=15, anchor="center")
    for df, file_name in zip(dataframes, file_names):
        info = display_dataset_info(df, file_name)
        Label(scrollable_frame, text=info, justify="center", font=("Arial", 12)).pack(pady=5)
    if not generation_lock:
        Button(scrollable_frame, text="Check for inconsistencies", command=lambda: show_inconsistencies(dataframes)).pack(pady=10)


    
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
    for df in dataframes:
            inconsistencies = detect_inconsistencies(df)
    if inconsistencies:
                messagebox.showwarning("Inconsistencies Found", f"Number of inconsistent groups: {len(inconsistencies)}")
                response=messagebox.askyesno("Inconsistencies Found","Do you want to remove inconsistencies?")
                if response:
                         remove_all_inconsistencies(dataframes)
                else:
                        return
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
    base64_icon = "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"  # Replace with actual Base64 string
    image_data = base64.b64decode(base64_icon)
    image = PhotoImage(data=image_data)
    root.iconphoto(True, image)
    Label(root, text="CSV Rule Analysis Tool", font=("Arial", 16)).pack(pady=20)
    Button(root, text="Select CSV Files", command=select_files, font=("Arial", 12)).pack(pady=30)
    Button(root, text="Dataset Info", command=show_dataset_info, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Generate decision rules", command=generate_decision_rules, font=("Arial", 12)).pack(pady=20)
    Button(root, text="Optimize Rules", command=select_files_optymalization, font=("Arial", 12)).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()
