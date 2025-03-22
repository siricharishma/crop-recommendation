import pandas as pd
import numpy as np
import argparse
import os
from model import ACSMModel

def predict_sample(model, snai, cfs):
    """Make a prediction for a single sample."""
    sample_input = pd.DataFrame({"SNAI": [snai], "CFS": [cfs]})
    
    # Model will handle the Weighted_Score calculation if needed
    predictions, distances = model.predict(sample_input, verbose=True)
    
    print("\nPREDICTION RESULTS")
    print("===================")
    print(f"Input: SNAI={snai}, CFS={cfs}")
    print(f"Predicted Crop: {predictions[0]}")
    
    # Get all distances for ranking
    all_distances = distances[0]
    sorted_crops = sorted(all_distances.items(), key=lambda x: x[1])
    
    print("\nCROP RECOMMENDATIONS (RANKED)")
    print("==============================")
    for i, (crop, distance) in enumerate(sorted_crops[:3], 1):
        print(f"{i}. {crop} (Distance: {distance:.4f})")
        
    return predictions[0], sorted_crops

def predict_batch(model, input_file, output_file=None):
    """Make predictions for a batch of samples from a CSV file."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None
    
    # Load input data
    print(f"Loading input data from {input_file}...")
    input_data = pd.read_csv(input_file)
    
    if "SNAI" not in input_data.columns or "CFS" not in input_data.columns:
        print("Error: Input file must contain SNAI and CFS columns.")
        return None
    
    # Make predictions
    predictions, _ = model.predict(input_data, verbose=True)
    
    # Add predictions to the dataframe
    output_data = input_data.copy()
    output_data["Predicted_Crop"] = predictions
    
    # Save results if output file is specified
    if output_file:
        output_data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    print(f"Processed {len(predictions)} samples.")
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Crop recommendation prediction tool")
    parser.add_argument("--model", default="acsm_model.joblib", help="Path to the trained model file")
    parser.add_argument("--mode", choices=["sample", "batch"], default="sample", help="Prediction mode: single sample or batch from CSV")
    parser.add_argument("--snai", type=float, help="SNAI value for single prediction")
    parser.add_argument("--cfs", type=float, help="CFS value for single prediction")
    parser.add_argument("--input", help="Input CSV file for batch prediction")
    parser.add_argument("--output", help="Output CSV file for batch prediction results")
    
    args = parser.parse_args()
    
    # Load the model
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    model = ACSMModel.load_model(args.model)
    
    # Process based on mode
    if args.mode == "sample":
        if args.snai is None or args.cfs is None:
            print("Error: SNAI and CFS values must be provided for sample mode.")
            return
        predict_sample(model, args.snai, args.cfs)
    else:  # batch mode
        if args.input is None:
            print("Error: Input file must be provided for batch mode.")
            return
        predict_batch(model, args.input, args.output)

if __name__ == "__main__":
    main()
