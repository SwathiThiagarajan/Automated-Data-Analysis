# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "requests",
#   "tenacity",
#   "tabulate"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

def load_data(filename):
    """Load CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filename, encoding='ISO-8859-1')
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform generic analysis: summary, missing values, correlations."""
    summary = df.describe(include='all')
    missing_values = df.isnull().sum()
    correlation_matrix = df.corr(numeric_only=True)
    return summary, missing_values, correlation_matrix

def generate_visualizations(df):
    """Generate and save enhanced PNG visualizations."""
    for col in df.select_dtypes(include=[np.number]).columns[:5]:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="royalblue")
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{col}.png", dpi=300)
        plt.close()

def get_openai_response(prompt):
    """Send prompt to OpenAI and return response, handling both HTML and JSON."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def chat():
        response = requests.post(url, headers=headers, json=data)
        print(f"Response Status Code: {response.status_code}")
        
        if "text/html" in response.headers.get('Content-Type', ''):
            print("Received HTML response, extracting text...")
            return response.text  # Return raw HTML content
        
        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except ValueError:
                print("Error parsing JSON response")
                print("Response Content:", response.text)
                return response.text  # Fallback to returning raw text
        else:
            raise Exception(f"Error: Received non-200 status code {response.status_code}")
    
    return chat()

def generate_report(filename, summary, missing_values, correlation_matrix, insights):
    """Create README.md with analysis results."""
    with open("README.md", "w") as f:
        f.write(f"# Automated Data Analysis Report\n\n")
        f.write(f"## Dataset: {filename}\n\n")
        f.write("### Summary Statistics\n\n")
        f.write(summary.to_markdown() + "\n\n")
        f.write("### Missing Values\n\n")
        f.write(missing_values.to_markdown() + "\n\n")
        f.write("### Correlation Matrix\n\n")
        f.write(correlation_matrix.to_markdown() + "\n\n")
        f.write("### AI-Generated Insights\n\n")
        f.write(insights + "\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = load_data(filename)
    summary, missing_values, correlation_matrix = analyze_data(df)
    generate_visualizations(df)
    
    prompt = f"""Analyze the following dataset:
    - Summary: {summary.to_string()}
    - Missing values: {missing_values.to_string()}
    - Correlation matrix: {correlation_matrix.to_string()}
    Provide insights as a story."""
    
    insights = get_openai_response(prompt)
    generate_report(filename, summary, missing_values, correlation_matrix, insights)
    print("Analysis complete. See README.md and PNG files.")

if __name__ == "__main__":
    main()
