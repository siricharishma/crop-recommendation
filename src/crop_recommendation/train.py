import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from crop_recommendation.model import ACSMModel, load_and_prepare_data, evaluate_model, plot_confusion_matrix

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
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
