import pandas as pd
import numpy as np
import os
from crop_recommendation.SNAI_calculator import calculate_snai
from crop_recommendation.CFS_calculator import calculate_cfs
import json

class DatasetPreparation:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.stats_file = os.path.join(model_dir, "reference_stats.json")
        self.snai_file = os.path.join(model_dir, "snai_data.csv")
        self.cfs_file = os.path.join(model_dir, "cfs_snai_data.csv")
        self.weighted_file = os.path.join(model_dir, "weighted_score_data.csv")
        
        # Initialize reference statistics
        self.reference_stats = {
            "N_mean": 0, "P_mean": 0, "K_mean": 0,
            "N_std": 0, "P_std": 0, "K_std": 0,
            "T_mean": 0, "H_mean": 0, "R_mean": 0,
            "T_std": 0, "H_std": 0, "R_std": 0
        }
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created model directory: {model_dir}")
    
    def load_raw_data(self, file_path):
        """Load the initial raw data file"""
        print(f"Loading raw data from {file_path}...")
        return pd.read_csv(file_path)
    
    def prepare_snai_data(self, raw_data_path):
        """Prepare SNAI data if it doesn't exist"""
        if os.path.exists(self.snai_file):
            print(f"Found existing SNAI data file: {self.snai_file}")
            return pd.read_csv(self.snai_file)
        
        print(f"Generating SNAI data from {raw_data_path}...")
        df = self.load_raw_data(raw_data_path)
        
        # Compute mean and standard deviation for N, P, K
        self.reference_stats["N_mean"] = df["N"].mean()
        self.reference_stats["P_mean"] = df["P"].mean() 
        self.reference_stats["K_mean"] = df["K"].mean()
        self.reference_stats["N_std"] = df["N"].std()
        self.reference_stats["P_std"] = df["P"].std()
        self.reference_stats["K_std"] = df["K"].std()
        
        # Apply function to calculate SNAI
        df["SNAI"] = df.apply(
            lambda row: calculate_snai(
                row["N"], row["P"], row["K"], 
                self.reference_stats["N_mean"], self.reference_stats["P_mean"], self.reference_stats["K_mean"],
                self.reference_stats["N_std"], self.reference_stats["P_std"], self.reference_stats["K_std"]
            ), axis=1
        )
        
        # Keep all original columns with SNAI
        df_snai = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type", "label", "SNAI"]]
        
        # Save the processed DataFrame
        df_snai.to_csv(self.snai_file, index=False)
        print(f"SNAI data saved to {self.snai_file}")
        
        return df_snai
    
    def prepare_cfs_data(self, snai_data=None):
        """Prepare CFS data if it doesn't exist"""
        if os.path.exists(self.cfs_file):
            print(f"Found existing CFS data file: {self.cfs_file}")
            return pd.read_csv(self.cfs_file)
        
        if snai_data is None:
            if not os.path.exists(self.snai_file):
                raise FileNotFoundError(f"SNAI data file not found: {self.snai_file}")
            snai_data = pd.read_csv(self.snai_file)
        
        print("Generating CFS data...")
        
        # Compute mean and standard deviation for temperature, humidity, and rainfall
        self.reference_stats["T_mean"] = snai_data["temperature"].mean()
        self.reference_stats["H_mean"] = snai_data["humidity"].mean()
        self.reference_stats["R_mean"] = snai_data["rainfall"].mean()
        self.reference_stats["T_std"] = snai_data["temperature"].std()
        self.reference_stats["H_std"] = snai_data["humidity"].std()
        self.reference_stats["R_std"] = snai_data["rainfall"].std()
        
        # Apply function to calculate CFS
        snai_data["CFS"] = snai_data.apply(
            lambda row: calculate_cfs(
                row["temperature"], row["humidity"], row["rainfall"],
                self.reference_stats["T_mean"], self.reference_stats["H_mean"], self.reference_stats["R_mean"],
                self.reference_stats["T_std"], self.reference_stats["H_std"], self.reference_stats["R_std"]
            ), axis=1
        )
        
        # Save the processed DataFrame
        snai_data.to_csv(self.cfs_file, index=False)
        print(f"CFS data saved to {self.cfs_file}")
        
        return snai_data
    
    def prepare_weighted_data(self, cfs_data=None):
        """Prepare weighted score data if it doesn't exist"""
        if os.path.exists(self.weighted_file):
            print(f"Found existing weighted data file: {self.weighted_file}")
            df = pd.read_csv(self.weighted_file)
            
            # Calculate weights from the data
            sigma_snai = df["SNAI"].std()
            sigma_cfs = df["CFS"].std()
            w1 = np.exp(sigma_cfs) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
            w2 = np.exp(sigma_snai) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
            
            # Save weights in reference stats
            self.reference_stats["w1"] = w1
            self.reference_stats["w2"] = w2
            
            return df, w1, w2
        
        if cfs_data is None:
            if not os.path.exists(self.cfs_file):
                raise FileNotFoundError(f"CFS data file not found: {self.cfs_file}")
            cfs_data = pd.read_csv(self.cfs_file)
        
        print("Generating weighted score data...")
        
        # Compute exponential weights
        sigma_snai = cfs_data["SNAI"].std()
        sigma_cfs = cfs_data["CFS"].std()
        w1 = np.exp(sigma_cfs) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        w2 = np.exp(sigma_snai) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        
        # Save weights in reference stats
        self.reference_stats["w1"] = w1
        self.reference_stats["w2"] = w2
        
        print(f"Exponential Weights - w1: {w1:.4f}, w2: {w2:.4f}")
        
        # Compute Weighted Score
        cfs_data["Weighted_Score"] = (w1 * cfs_data["SNAI"]) + (w2 * cfs_data["CFS"])
        
        # Save the dataset
        cfs_data.to_csv(self.weighted_file, index=False)
        print(f"Weighted score data saved to {self.weighted_file}")
        
        return cfs_data, w1, w2
    
    def save_reference_stats(self):
        """Save reference statistics to JSON file"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.reference_stats, f, indent=4)
        print(f"Reference statistics saved to {self.stats_file}")
    
    def load_reference_stats(self):
        """Load reference statistics from JSON file"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.reference_stats = json.load(f)
            print(f"Loaded reference statistics from {self.stats_file}")
            return self.reference_stats
        else:
            print(f"Reference statistics file not found: {self.stats_file}")
            return None
    
    def prepare_dataset(self, raw_data_path="data.csv"):
        """Prepare the complete dataset with all features"""
        # Prepare SNAI data
        snai_data = self.prepare_snai_data(raw_data_path)
        
        # Prepare CFS data
        cfs_data = self.prepare_cfs_data(snai_data)
        
        # Prepare weighted data
        weighted_data, w1, w2 = self.prepare_weighted_data(cfs_data)
        
        # Save reference statistics
        self.save_reference_stats()
        
        # Return features and target
        X = weighted_data[["SNAI", "CFS", "Weighted_Score"]]
        y = weighted_data["label"]
        
        return X, y, w1, w2

def prepare_data(model_dir="model", raw_data_path="data.csv"):
    """Helper function to prepare data and get reference statistics"""
    data_prep = DatasetPreparation(model_dir)
    X, y, w1, w2 = data_prep.prepare_dataset(raw_data_path)
    reference_stats = data_prep.reference_stats
    
    return X, y, w1, w2, reference_stats

if __name__ == "__main__":
    # Set model directory
    model_dir = "model"
    
    # Load and prepare data with the dataset module
    print("Preparing dataset...")
    X, y, w1, w2, reference_stats = prepare_data(model_dir=model_dir)
    print("Dataset preparation complete!")
