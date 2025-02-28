# Automated Data Analysis & Storytelling

This project implements an automated data analysis tool using Python and an LLM (specifically, GPT-4o-Mini). The goal is to take any CSV dataset, perform a comprehensive exploratory analysis, visualize key insights, and finally, narrate a data story in a Markdown file.

## Problem Statement

In today's data-driven world, gaining insights from raw data is essential. This project challenges you to create a Python script that:

1. **Analyzes Data:**  
   - Performs generic analysis on any CSV dataset by computing summary statistics, detecting missing values, uncovering correlations, spotting outliers, and exploring other analytical avenues (clustering, time series, geographic, network analysis, etc.).
   - Dynamically adapts to the structure of the input CSV without assuming a fixed format.

2. **Visualizes Results:**  
   - Generates 1-3 supporting data visualizations (e.g., charts or heatmaps) using Python libraries such as Seaborn or Matplotlib.
   - Saves each chart as a PNG image file.

3. **Narrates a Story:**  
   - Uses an LLM to synthesize the analysis into a coherent narrative.
   - Creates a README.md file that tells the story of the data, detailing:
     - A brief description of the input data.
     - The analysis performed.
     - The key insights discovered.
     - The implications of the findings.
