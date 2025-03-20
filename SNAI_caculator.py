import pandas as pd

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to compute Soil Nutrient Adaptive Index (SNAI) using mean and standard deviation
def calculate_snai(N, P, K, N_mean, P_mean, K_mean, N_std, P_std, K_std):
    deviation = abs(N - N_mean) / N_std + abs(P - P_mean) / P_std + abs(K - K_mean) / K_std
    snai = 100 - ((deviation / 3) * 100)  # Normalizing over 3 parameters
    return round(snai, 2)

# Main execution
if __name__ == "__main__":
    file_path = r"C:\Users\charishma\Desktop\Major Project\data.csv"  # Change this to your actual file path
    df = load_data(file_path)
    
    # Compute mean and standard deviation for N, P, K
    N_mean, P_mean, K_mean = df["N"].mean(), df["P"].mean(), df["K"].mean()
    N_std, P_std, K_std = df["N"].std(), df["P"].std(), df["K"].std()
    
    # Apply function to calculate SNAI
    df["SNAI"] = df.apply(lambda row: calculate_snai(row["N"], row["P"], row["K"], N_mean, P_mean, K_mean, N_std, P_std, K_std), axis=1)
    
  
    
    # Group by Soil_Type and Label, and compute mean SNAI
    df_grouped = df.groupby(["soil_type", "label"], as_index=False).mean()
    
    # Keep all original columns except N, P, K
    df_grouped = df[["N","P","K","temperature", "humidity", "ph", "rainfall", "soil_type", "label", "SNAI"]]
    
    # Save the processed DataFrame to a new CSV file
    df_grouped.to_csv("snai_processed_data_1.csv", index=False)
    
    # Display results
    print(df_grouped)
