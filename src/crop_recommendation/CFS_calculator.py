import pandas as pd

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to compute Climate Flexibility Score (CFS) with adjustments
def calculate_cfs(temp, humidity, rainfall, T_mean, H_mean, R_mean, T_std, H_std, R_std):
    if T_std == 0: T_std = 1  # Avoid division by zero
    if H_std == 0: H_std = 1
    if R_std == 0: R_std = 1

    deviation = ((temp - T_mean) / T_std) ** 2 + ((humidity - H_mean) / H_std) ** 2 + ((rainfall - R_mean) / R_std) ** 2
    cfs = 100 - ((deviation / 3) * 100)  # Normalizing over 3 parameters
    return round(max(0, cfs), 2)  # Ensure CFS is not negative

# Main execution
if __name__ == "__main__":
    file_path = r"snai_data.csv"  # Change this to your actual file path
    df = load_data(file_path)
    
    # Compute mean and standard deviation for temperature, humidity, and rainfall
    T_mean, H_mean, R_mean = df["temperature"].mean(), df["humidity"].mean(), df["rainfall"].mean()
    T_std, H_std, R_std = df["temperature"].std(), df["humidity"].std(), df["rainfall"].std()
    
    # Apply function to calculate CFS
    df["CFS"] = df.apply(lambda row: calculate_cfs(row["temperature"], row["humidity"], row["rainfall"], T_mean, H_mean, R_mean, T_std, H_std, R_std), axis=1)


    # Save the CFS data into a separate CSV file
    df.to_csv("cfs_snai_data.csv", index=False)

    # Display results
    print(df)
