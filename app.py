import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from typing import List

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Step 1: Loading and verifying data
def load_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """Load multiple CSV files into a list of DataFrames."""
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
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
    """Funkcja df.iterrows() jest Heurystyką RM, gdzie dla każdego wiersza generowana jest jedna reguła decyzyjna, bazująca na wartości wskazanego wiersza iteracji."""
    for _, row in df.iterrows(): 
        condition = {col: int(row[col]) if isinstance(row[col], (np.integer, np.int64)) else row[col] for col in df.columns[:-1] if pd.notna(row[col])}
        decision = row[df.columns[-1]]
        rules.append({"condition": condition, "decision": decision})
    return rules

# Step 3: Calculate rule lengths and optimize
def calculate_rule_length(rule: dict) -> int:
    return len(rule["condition"])

def optimize_rules(rules: List[dict]) -> List[dict]:
    """Optymalizacja reguły pod zwględem długości (wartość minimalna)"""
    optimized_rules = []
    for rule in rules:
        length = calculate_rule_length(rule)
        optimized_rules.append({**rule, "length": length})
    return sorted(optimized_rules, key=lambda r: r["length"])

# Step 4: Analyze results
def analyze_rules(rules: List[dict]) -> dict:
    """Analyze rules to calculate statistics (unique rules, min/avg/max lengths)."""
    unique_rules = {frozenset(rule["condition"].items()): rule for rule in rules}
    lengths = [rule["length"] for rule in unique_rules.values()]

    return {
        "unique_rules_count": len(unique_rules),
        "min_length": min(lengths),
        "avg_length": np.mean(lengths),
        "max_length": max(lengths),
        "unique_rules": list(unique_rules.values()),
    }

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return redirect(request.url)

        file_paths = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)

        # Process files
        dataframes = load_csv_files(file_paths)
        all_rules = []

        for df in dataframes:
            verified_df = verify_data(df)
            rules = induce_rules(verified_df)
            optimized_rules = optimize_rules(rules)
            all_rules.extend(optimized_rules)

        analysis = analyze_rules(all_rules)

        # Generate plot
        fig, ax = plt.subplots()
        ax.bar(['Min Length', 'Avg Length', 'Max Length'], [
            analysis['min_length'], 
            analysis['avg_length'], 
            analysis['max_length']
        ])
        ax.set_title('Rule Length Analysis')
        ax.set_ylabel('Length')

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return render_template('result.html', analysis=analysis, plot_data=plot_data)

    return render_template('index.html')

# HTML Templates
index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Global Optimization</title>
</head>
<body>
    <h1>Upload CSV Files</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple>
        <input type="submit" value="Upload">
    </form>
</body>
</html>"""

result_html = """<!DOCTYPE html>
<html>
<head>
    <title>Results</title>
</head>
<body>
    <h1>Verification and Rule Analysis</h1>

    <h2>Analysis</h2>
    <ul>
        <li>Unique Rules Count: {{ analysis['unique_rules_count'] }}</li>
        <li>Min Length: {{ analysis['min_length'] }}</li>
        <li>Avg Length: {{ analysis['avg_length'] }}</li>
        <li>Max Length: {{ analysis['max_length'] }}</li>
    </ul>

    <h2>Unique Rules</h2>
    <ul>
        {% for rule in analysis['unique_rules'] %}
        <li>Condition: {{ rule['condition'] }}, Decision: {{ rule['decision'] }}, Length: {{ rule['length'] }}</li>
        {% endfor %}
    </ul>

    <h2>Rule Length Analysis</h2>
    <img src="data:image/png;base64,{{ plot_data }}">

    <a href="/">Upload More Files</a>
</body>
</html>"""

# Save templates to files
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(index_html)

with open('templates/result.html', 'w') as f:
    f.write(result_html)

if __name__ == "__main__":
    app.run(debug=True)
