import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from crop_recommendation.model import ACSMModel
from crop_recommendation.SNAI_calculator import calculate_snai
from crop_recommendation.CFS_calculator import calculate_cfs

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(model_path="model/acsm_model.joblib"):
    """Load the trained model with caching for efficiency"""
    try:
        model = ACSMModel.load_model(model_path)
        return model, True
    except Exception as e:
        return str(e), False

def calculate_weighted_score(snai, cfs, w1, w2):
    """Calculate the weighted score based on SNAI and CFS"""
    return (w1 * snai) + (w2 * cfs)

def predict_for_input(model, snai, cfs):
    """Make a prediction for single input"""
    sample_input = pd.DataFrame({"SNAI": [snai], "CFS": [cfs]})
    predictions, distances = model.predict(sample_input)
    
    # Get all distances for ranking
    all_distances = distances[0]
    sorted_crops = sorted(all_distances.items(), key=lambda x: x[1])
    
    return predictions[0], sorted_crops

def plot_crop_rankings(crop_rankings, top_n=5):
    """Create a bar chart of crop rankings"""
    top_crops = crop_rankings[:top_n]
    crops = [crop for crop, _ in top_crops]
    distances = [distance for _, distance in top_crops]
    
    # Create a color gradient - lower distance (better) gets darker color
    colors = sns.color_palette("YlGn_r", len(crops))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(crops, distances, color=colors)
    
    ax.set_title('Top Crop Recommendations by Distance', fontsize=16)
    ax.set_xlabel('Crop', fontsize=14)
    ax.set_ylabel('Mahalanobis Distance (lower is better)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig

def run_app():
    """Main function to run the Streamlit app"""
    st.title("🌱 Crop Recommendation System")
    
    # Sidebar with app information
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses SNAI  and CFS "
            " to recommend suitable crops for given conditions."
        )
        
        st.header("How it works")
        st.write(
            "1. Enter soil nutrient (N, P, K) and climate data (temperature, humidity, rainfall)\n"
            "2. The app calculates SNAI and CFS scores\n"
            "3. The model predicts the most suitable crops for your conditions"
        )
        
        st.header("Model Information")
        model_result = load_model()
        if model_result[1]:  # If model loaded successfully
            model = model_result[0]
            st.success("✅ Model loaded successfully!")
            if hasattr(model, 'w1') and hasattr(model, 'w2'):
                st.write(f"Model weights: w1={model.w1:.4f}, w2={model.w2:.4f}")
            if hasattr(model, 'crop_classes'):
                st.write(f"Model trained on {len(model.crop_classes)} crop classes")
        else:
            st.error(f"❌ Failed to load model: {model_result[0]}")
            st.stop()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    # Single Prediction Tab
    with tab1:
        st.header("Enter Soil and Climate Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            n_value = st.number_input("Nitrogen (N) mg/kg", min_value=0, max_value=1000, value=90)
            p_value = st.number_input("Phosphorus (P) mg/kg", min_value=0, max_value=1000, value=42)
            k_value = st.number_input("Potassium (K) mg/kg", min_value=0, max_value=1000, value=43)
            ph_value = st.slider("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            
            # Get reference statistics from the model
            model = model_result[0]
            ref_stats = model.get_reference_stats()
            
            # Use reference statistics from the model if available, otherwise use defaults
            if ref_stats and all(k in ref_stats for k in ["N_mean", "P_mean", "K_mean", "N_std", "P_std", "K_std"]):
                snai_ref = {
                    "N_mean": ref_stats["N_mean"], 
                    "P_mean": ref_stats["P_mean"], 
                    "K_mean": ref_stats["K_mean"],
                    "N_std": ref_stats["N_std"], 
                    "P_std": ref_stats["P_std"], 
                    "K_std": ref_stats["K_std"]
                }
            else:
                # Fallback to default values if model doesn't have reference stats
                snai_ref = {
                    "N_mean": 50.55, "P_mean": 53.36, "K_mean": 48.15,
                    "N_std": 36.92, "P_std": 33.69, "K_std": 50.34
                }
            
            snai = calculate_snai(
                n_value, p_value, k_value, 
                snai_ref["N_mean"], snai_ref["P_mean"], snai_ref["K_mean"],
                snai_ref["N_std"], snai_ref["P_std"], snai_ref["K_std"]
            )
            
            st.metric("SNAI Score", f"{snai:.2f}")
        
        with col2:
            st.subheader("Climate Parameters")
            temp_value = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=20.87)
            humidity_value = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
            rainfall_value = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=202.93)
            
            # Use reference statistics from the model if available, otherwise use defaults
            if ref_stats and all(k in ref_stats for k in ["T_mean", "H_mean", "R_mean", "T_std", "H_std", "R_std"]):
                cfs_ref = {
                    "T_mean": ref_stats["T_mean"], 
                    "H_mean": ref_stats["H_mean"], 
                    "R_mean": ref_stats["R_mean"],
                    "T_std": ref_stats["T_std"], 
                    "H_std": ref_stats["H_std"], 
                    "R_std": ref_stats["R_std"]
                }
            else:
                # Fallback to default values if model doesn't have reference stats
                cfs_ref = {
                    "T_mean": 25.62, "H_mean": 71.48, "R_mean": 103.46,
                    "T_std": 5.06, "H_std": 22.26, "R_std": 54.96
                }
            
            cfs = calculate_cfs(
                temp_value, humidity_value, rainfall_value,
                cfs_ref["T_mean"], cfs_ref["H_mean"], cfs_ref["R_mean"],
                cfs_ref["T_std"], cfs_ref["H_std"], cfs_ref["R_std"]
            )
            
            st.metric("CFS Score", f"{cfs:.2f}")
        
        # Prediction section
        st.subheader("Prediction")
        if st.button("Get Crop Recommendations", type="primary"):
            with st.spinner("Analyzing data and generating recommendations..."):
                model = model_result[0]  # Get the loaded model
                
                # Make prediction
                best_crop, crop_rankings = predict_for_input(model, snai, cfs)
                
                # Calculate weighted score for display
                weighted_score = calculate_weighted_score(snai, cfs, model.w1, model.w2)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("Top Recommendation", best_crop)
                col2.metric("SNAI", f"{snai:.2f}")
                col3.metric("CFS", f"{cfs:.2f}")
                
                st.write(f"Weighted Score: {weighted_score:.2f} (w1={model.w1:.4f}, w2={model.w2:.4f})")
                
                # Display visualization
                st.write("### Crop Ranking")
                fig = plot_crop_rankings(crop_rankings)
                st.pyplot(fig)
                
                # Display top 5 recommendations in a table
                st.write("### Top 5 Recommendations")
                top5_df = pd.DataFrame(
                    crop_rankings[:5], 
                    columns=["Crop", "Distance Score"]
                )
                top5_df.index = range(1, len(top5_df) + 1)  # 1-based indexing
                top5_df["Distance Score"] = top5_df["Distance Score"].round(4)
                st.table(top5_df)
                
                # Explanation
                st.info(
                    "The Mahalanobis distance measures how far your soil and climate "
                    "conditions are from the typical conditions for each crop. "
                    "Lower distances indicate better matches."
                )
    
    # Batch Prediction Tab
    with tab2:
        st.header("Batch Prediction")
        st.write(
            "Upload a CSV file with SNAI and CFS columns to get crop recommendations "
            "for multiple samples at once."
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Load data from uploaded file
            try:
                input_data = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview:")
                st.dataframe(input_data.head())
                
                # Check if required columns exist
                if "SNAI" not in input_data.columns or "CFS" not in input_data.columns:
                    st.error("Error: Input file must contain SNAI and CFS columns.")
                else:
                    if st.button("Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            # Get model
                            model = model_result[0]
                            
                            # Make predictions
                            predictions, _ = model.predict(input_data)
                            
                            # Add predictions to the dataframe
                            output_data = input_data.copy()
                            output_data["Predicted_Crop"] = predictions
                            
                            # Display results
                            st.write("### Prediction Results")
                            st.dataframe(output_data)
                            
                            # Offer download option
                            csv = output_data.to_csv(index=False)
                            st.download_button(
                                label="Download predictions as CSV",
                                data=csv,
                                file_name="crop_predictions.csv",
                                mime="text/csv",
                            )
                            
                            # Show distribution of predictions
                            st.write("### Distribution of Predicted Crops")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            crop_counts = output_data["Predicted_Crop"].value_counts()
                            crop_counts.plot(kind="bar", ax=ax)
                            ax.set_title("Number of Samples per Predicted Crop")
                            ax.set_xlabel("Crop")
                            ax.set_ylabel("Count")
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    run_app()
