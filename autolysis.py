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
        df = pd.read_csv(filename ,encoding='ISO-8859-1')
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
    """Generate and save PNG visualizations."""
    for col in df.select_dtypes(include=[np.number]).columns[:3]:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{col}.png")
        plt.close()

def get_together_ai_response(prompt):
    """Send prompt to Together AI and return response."""
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        print("Error: TOGETHER_API_KEY not set.")
        sys.exit(1)
    
    url = "https://api.together.ai/playground/chat/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    headers = {"Authorization": f"Bearer {together_api_key}", "Content-Type": "application/json"}
    data = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def chat():
        response = requests.post(url, headers=headers, json=data)
    
        # Log the response status code and text for debugging
        print(f"Response Status Code: {response.status_code}")
        
        # Check for HTML response
        if "html" in response.headers.get('Content-Type', ''):
            print("Received HTML response, possibly an error page.")
            return f"Error: Received HTML response instead of expected JSON. Status Code: {response.status_code}"
        
        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except ValueError:
                print("Error parsing JSON response")
                print("Response Content:", response.text)  # Log full response content if not JSON
                raise
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
    
    insights = get_together_ai_response(prompt)
    generate_report(filename, summary, missing_values, correlation_matrix, insights)
    print("Analysis complete. See README.md and PNG files.")

if __name__ == "__main__":
    main()
