import pandas as pd
import re

def extract_templates(input_csv, output_csv):
    """
    Extract unique templates from the gluecontrol log file and save them in a new CSV file.
    """
    # Read the input CSV file
    data = pd.read_csv(input_csv, sep='->', names=['EventId', 'EventTemplate'], engine='python')

    # Extract unique templates
    unique_templates = data.drop_duplicates(subset=['EventTemplate'])
    df_filtered = unique_templates[~unique_templates['EventTemplate'].str.contains('弯翘判定模块', na=False)]

    # Save the unique templates to the output CSV file
    df_filtered.to_csv(output_csv, index=False, sep='@')

if __name__ == "__main__":
    input_file = "log_data/ips.csv"
    output_file = "log_data/ips_templates.csv"
    extract_templates(input_file, output_file)
    print(f"Templates extracted and saved to {output_file}")