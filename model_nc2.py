import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import mahalanobis

# Load SNAI and CFS data from CSV files
print("Loading data...")
snai_df = pd.read_csv(r"C:\Users\charishma\Desktop\Major Project\utils\snai.csv")
cfs_df = pd.read_csv(r"C:\Users\charishma\Desktop\Major Project\utils\cfs_data.csv")
print("Data loaded successfully!")

# Merge datasets on common columns like 'soil_type' and 'label'
print("Merging datasets...")
merged_df = pd.merge(snai_df, cfs_df, on=["soil_type", "label"], how="inner")
print("Datasets merged successfully!")

# Compute exponential weights for dynamic weighting
print("Calculating exponential weights...")
sigma_snai = merged_df["SNAI"].std()
sigma_cfs = merged_df["CFS"].std()
w1 = np.exp(sigma_cfs) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
w2 = np.exp(sigma_snai) / (np.exp(sigma_snai) + np.exp(sigma_cfs))
print(f"Exponential Weights - w1: {w1}, w2: {w2}")

# Compute Exponential Weighted Score (EWS)
merged_df["Weighted_Score"] = (w1 * merged_df["SNAI"]) + (w2 * merged_df["CFS"])
print("Weighted Score calculated!")

# Save the dataset with EWS (_2 suffix)
merged_df.to_csv("weighted_score_data_nc2.csv", index=False)
print("Weighted score data saved!")

# Features and target variable
X = merged_df[["SNAI", "CFS", "Weighted_Score"]]
y = merged_df["label"]

# Split the dataset
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully!")

# Custom ACSM model with Mahalanobis distance
class ACSMModel:
    def __init__(self):
        self.crop_classes = {}
        self.cov_matrix_inv = None

    def fit(self, X, y):
        print("Training model...")
        for crop in y.unique():
            self.crop_classes[crop] = X[y == crop].mean().to_dict()
        self.cov_matrix_inv = np.linalg.inv(np.cov(X.T))
        print("Model trained successfully!")

    def predict(self, X):
        print("Making predictions...")
        predictions = []
        for _, row in X.iterrows():
            best_crop = None
            min_distance = float("inf")
            for crop, values in self.crop_classes.items():
                diff = np.array([row[feature] - values[feature] for feature in ["SNAI", "CFS", "Weighted_Score"]])
                distance = mahalanobis(diff, np.zeros_like(diff), self.cov_matrix_inv)
                if distance < min_distance:
                    min_distance = distance
                    best_crop = crop
            predictions.append(best_crop)
        print("Predictions completed!")
        return predictions

# Train the model
acsm = ACSMModel()
acsm.fit(X_train, y_train)

# Predict for the test data
y_pred = acsm.predict(X_test)

# Save the predictions (_2 suffix)
predictions_df = X_test.copy()
predictions_df["Predicted_Crop"] = y_pred
predictions_df.to_csv("output_nc2.csv", index=False)
print("✅ Predictions saved!")

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print("\n📊 **Performance Metrics**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")

# Test a Sample Input
sample_input = pd.DataFrame({"SNAI": [10.5], "CFS": [30.2], "Weighted_Score": [w1 * 10.5 + w2 * 30.2]})
sample_prediction = acsm.predict(sample_input)

print("\n🌾 **Test Sample Prediction**")
print(f"Input Values: {sample_input.to_dict(orient='records')[0]}")
print(f"Predicted Crop: {sample_prediction[0]}")

# Next Best Crop Calculation
def get_next_best_crop(X, first_crop):
    filtered_crops = {crop: values for crop, values in acsm.crop_classes.items() if crop != first_crop}
    min_distance = float("inf")
    next_best_crop = None
    for crop, values in filtered_crops.items():
        diff = np.array([X[feature].values[0] - values[feature] for feature in ["SNAI", "CFS", "Weighted_Score"]])
        distance = mahalanobis(diff, np.zeros_like(diff), acsm.cov_matrix_inv)
        if distance < min_distance:
            min_distance = distance
            next_best_crop = crop
    return next_best_crop

next_best_crop = get_next_best_crop(sample_input, sample_prediction[0])
print(f"Next Best Crop: {next_best_crop}")
