import pandas as pd
import numpy as np
from io import StringIO
import glob

def calculate_map(df, difficulties):
    """Calculates mAP by averaging precision values."""
    results = {}
    for diff in difficulties:
        if diff in df.columns:
            results[diff] = df[diff].mean()
    return results

def process_data(data_string, header=None):
    """Processes a data string and returns a DataFrame."""
    try:
        df = pd.read_csv(StringIO(data_string), sep=r'\s+', header=header)
        df.columns = ['Recall', 'Easy', 'Moderate', 'Hard']
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def process_file(file_path):
    """Reads a file, calculates mAP, and returns a dictionary."""
    with open(file_path, 'r') as f:
        file_data = f.read()
    
    df = process_data(file_data)
    if df is not None:
        return calculate_map(df, ['Easy', 'Moderate', 'Hard'])
    return None

def print_results(file_name, results):
    """Prints the mAP results in a formatted table."""
    print(f"\nResults for {file_name}:")
    table = pd.DataFrame([results]).T
    table.columns = ['mAP']
    table.index.name = 'Difficulty'
    print(table.to_markdown(floatfmt=".4f"))



# --- Process all files in results/plot/ directory ---
print("\n" + "="*50)
print("Processing files in 'results/plot/' directory:")

# Check if the directory exists
plot_dir = 'results/plot-new-data-full-s-New-3layer'  # Adjust this to your directory
import os
if not os.path.exists(plot_dir):
    print(f"Directory '{plot_dir}' not found.")
else:
    file_list = glob.glob(os.path.join(plot_dir, '*.txt'))
    if not file_list:
        print(f"No .txt files found in '{plot_dir}'.")
    else:
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            file_results = process_file(file_path)
            if file_results is not None:
                print_results(file_name, file_results)