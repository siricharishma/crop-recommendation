import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, ConfusionMatrixDisplay)
from scipy.spatial.distance import mahalanobis
import os

# Data loading and preparation functions
def load_and_prepare_data():
    # Check if weighted score data already exists
    weighted_data_path = "weighted_score_data.csv"
    if os.path.exists(weighted_data_path):
        print(f"Found existing weighted data file: {weighted_data_path}")
        print("Loading pre-computed weighted data...")
        
        # Load the pre-computed data
        merged_df = pd.read_csv(weighted_data_path)
        
        # Calculate w1 and w2 from the data to maintain consistency
        sigma_snai = merged_df["SNAI"].std()
        sigma_cfs = merged_df["CFS"].std()
        w1 = np.exp(sigma_cfs) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        w2 = np.exp(sigma_snai) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        
        print(f"Loaded weighted data: {len(merged_df)} samples")
        print(f"Using weights - w1: {w1:.4f}, w2: {w2:.4f}")
    else:
        print("No existing weighted data found. Processing raw data...")
        # Original data loading and processing code
        print("Loading data...")
        merged_df = pd.read_csv(r"cfs_snai_data.csv")
        print("Data loaded successfully!")

        # Compute exponential weights
        print("Calculating exponential weights...")
        sigma_snai = merged_df["SNAI"].std()
        sigma_cfs = merged_df["CFS"].std()
        w1 = np.exp(sigma_cfs) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        w2 = np.exp(sigma_snai) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
        print(f"Exponential Weights - w1: {w1:.4f}, w2: {w2:.4f}")

        # Compute Weighted Score
        merged_df["Weighted_Score"] = (w1 * merged_df["SNAI"]) + (w2 * merged_df["CFS"])
        print("Weighted Score calculated!")

        # Save the dataset
        merged_df.to_csv(weighted_data_path, index=False)
        print(f"Weighted score data saved to {weighted_data_path}!")

    # Features and target
    X = merged_df[["SNAI", "CFS", "Weighted_Score"]]
    y = merged_df["label"]
    
    return X, y, w1, w2

# Custom ACSM model with Mahalanobis distance
class ACSMModel:
    def __init__(self):
        self.crop_classes = {}
        self.cov_matrix_inv = None
        self.training_time = None
        self.features = ["SNAI", "CFS", "Weighted_Score"]
        self.w1 = None
        self.w2 = None

    def fit(self, X, y, w1=None, w2=None, save_path=None, verbose=True):
        """Train the ACSM model with detailed progress information."""
        start_time = time.time()
        
        # Store weights for later predictions
        self.w1 = w1
        self.w2 = w2
        
        if verbose:
            print("\nTraining ACSM model...")
            print(f"Dataset: {len(X)} samples, {len(y.unique())} unique crops")
        
        # Calculate centroids for each crop class
        class_counts = {}
        for i, crop in enumerate(y.unique()):
            crop_data = X[y == crop]
            class_counts[crop] = len(crop_data)
            self.crop_classes[crop] = crop_data.mean().to_dict()
            if verbose:
                print(f"  Class {i+1}/{len(y.unique())}: '{crop}' ({len(crop_data)} samples)")
        
        # Calculate inverse covariance matrix
        if verbose:
            print("Computing covariance matrix...")
        self.cov_matrix_inv = np.linalg.inv(np.cov(X.T))
        
        # Record training time
        self.training_time = time.time() - start_time
        if verbose:
            print(f"Model trained successfully in {self.training_time:.2f} seconds!")
            print(f"Class distribution: {class_counts}")
        
        # Save model if path provided
        if save_path:
            if verbose:
                print(f"Saving model to {save_path}...")
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                if verbose:
                    print(f"Created directory: {save_dir}")
            joblib.dump(self, save_path)
            if verbose:
                print("Model saved successfully!")
        
        return self
    
    @classmethod
    def load_model(cls, filepath):
        """Load a previously saved model."""
        print(f"Loading model from {filepath}...")
        model = joblib.load(filepath)
        print("Model loaded successfully!")
        return model

    def predict(self, X, verbose=True):
        """Make predictions for input data."""
        if verbose:
            print("Making predictions...")
        
        # If input doesn't have Weighted_Score and we have weights, calculate it
        if "Weighted_Score" not in X.columns and self.w1 is not None and self.w2 is not None:
            X = X.copy()  # Avoid modifying the original
            X["Weighted_Score"] = (self.w1 * X["SNAI"]) + (self.w2 * X["CFS"])
            if verbose:
                print(f"Added Weighted_Score using w1={self.w1:.4f}, w2={self.w2:.4f}")
            
        predictions = []
        distances = {}  # Store distances for each sample and crop
        
        for idx, row in X.iterrows():
            best_crop = None
            min_distance = float("inf")
            sample_distances = {}
            
            for crop, values in self.crop_classes.items():
                diff = np.array([row[feature] - values[feature] for feature in self.features])
                distance = mahalanobis(diff, np.zeros_like(diff), self.cov_matrix_inv)
                sample_distances[crop] = distance
                
                if distance < min_distance:
                    min_distance = distance
                    best_crop = crop
                    
            predictions.append(best_crop)
            distances[idx] = sample_distances
            
        if verbose:
            print("Predictions completed!")
            
        return predictions, distances

# Evaluation functions
def evaluate_model(y_true, y_pred, output_file=None):
    """Calculate and display performance metrics."""
    print("\nPERFORMANCE METRICS")
    print("=====================")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        print(f"Metrics saved to {output_file}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Generate and display confusion matrix."""
    print("\nGenerating confusion matrix...")
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def test_sample_prediction(model, sample_data, w1, w2):
    """Test prediction on a sample input."""
    print("\nSAMPLE PREDICTION TEST")
    print("======================")
    
    # Ensure sample data has the weighted score
    if "Weighted_Score" not in sample_data.columns:
        sample_data["Weighted_Score"] = (w1 * sample_data["SNAI"]) + (w2 * sample_data["CFS"])
    
    sample_pred, distances = model.predict(sample_data)
    
    print(f"Input Values: {sample_data.to_dict(orient='records')[0]}")
    print(f"Predicted Crop: {sample_pred[0]}")
    
    # Calculate next best crop
    all_distances = distances[0]  # Get distances for the first (only) sample
    sorted_crops = sorted(all_distances.items(), key=lambda x: x[1])
    
    print("\nCROP RECOMMENDATIONS (RANKED)")
    print("==============================")
    for i, (crop, distance) in enumerate(sorted_crops[:3], 1):
        print(f"{i}. {crop} (Distance: {distance:.4f})")
    
    return sample_pred[0], sorted_crops

# Main execution function
def main():
    # Load and prepare data
    X, y, w1, w2 = load_and_prepare_data()
    
    # Split the dataset
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successfully! (Train: {len(X_train)}, Test: {len(X_test)})")
    
    # Train and save model
    model_path = "acsm_model.joblib"
    acsm = ACSMModel()
    acsm.fit(X_train, y_train, w1=w1, w2=w2, save_path=model_path)
    
    # Make predictions
    y_pred, _ = acsm.predict(X_test)
    
    # Save the predictions
    predictions_df = X_test.copy()
    predictions_df["Predicted_Crop"] = y_pred
    predictions_df.to_csv("output_nc2.csv", index=False)
    print("Predictions saved to output_nc2.csv!")
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, output_file="model_metrics.txt")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png")
    
    # Test sample prediction
    sample_input = pd.DataFrame({"SNAI": [10.5], "CFS": [30.2]})
    best_crop, crop_rankings = test_sample_prediction(acsm, sample_input, w1, w2)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()