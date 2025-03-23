import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from crop_recommendation.model import ACSMModel
from crop_recommendation.dataset import prepare_data
from crop_recommendation.model import evaluate_model, plot_confusion_matrix

def main():
    # Set model directory
    model_dir = "model"
    
    # Load and prepare data with the dataset module
    print("Preparing dataset...")
    X, y, w1, w2, reference_stats = prepare_data(model_dir=model_dir)
    print("Dataset preparation complete!")
    
    # Split the dataset
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successfully! (Train: {len(X_train)}, Test: {len(X_test)})")
    
    # Train and save model
    model_path = f"{model_dir}/acsm_model.joblib"
    acsm = ACSMModel()
    acsm.fit(X_train, y_train, w1=w1, w2=w2, reference_stats=reference_stats, save_path=model_path)
    
    # Make predictions
    y_pred, _ = acsm.predict(X_test)
    
    # Save the predictions
    predictions_df = X_test.copy()
    predictions_df["Predicted_Crop"] = y_pred
    predictions_df.to_csv(f"{model_dir}/output_predictions.csv", index=False)
    print(f"Predictions saved to {model_dir}/output_predictions.csv!")
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, output_file=f"{model_dir}/metrics.txt")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path=f"{model_dir}/confusion_matrix.png")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
