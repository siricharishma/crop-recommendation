import pandas as pd

# Load the dataset from CSV
df = pd.read_csv(r"cfs_data.csv")  # Load the previously saved CFS data

# Define categories for CFS
def categorize_cfs(value):
    if value < 50:
        return "Low"
    elif 50 <= value < 75:
        return "Medium"
    else:
        return "High"

# Apply categorization and create a new column
df["CFS_Category"] = df["CFS"].apply(categorize_cfs)



# Save the updated DataFrame to a new CSV file
df.to_csv("categorized_cfs_data.csv", index=False)

# Display the first few rows of the updated DataFrame
print(df.head())
