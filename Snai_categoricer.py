import pandas as pd

# Load the dataset from CSV
file_path = r"C:\Users\charishma\Desktop\Major Project\utils\snai_processed_data_1.csv"  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Ensure "SNAI" column exists before processing
if "SNAI" in df.columns:
    # Define categories for SNAI
    def categorize_snai(value):
        if value < 50:
            return "Low"
        elif 50 <= value < 75:
            return "Medium"
        else:
            return "High"

    # Apply categorization and replace the SNAI column
    df["SNAI"] = df["SNAI"].apply(categorize_snai)

    # Save the updated DataFrame to a new CSV file
    df.to_csv("categorised_snai_data.csv", index=False)  # Fixed the syntax error here

    # Display the first few rows of the updated DataFrame
    print(df.head())
else:
    print("Error: 'SNAI' column not found in the dataset.")
