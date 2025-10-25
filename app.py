import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- Load the pre-trained objects ---
try:
    pt = joblib.load('power_transformer.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_5_components.pkl') # Keep loading PCA in case you want to display components later
    model_rf = joblib.load('random_forest_model.pkl')
    print("Models and preprocessors loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure the .pkl files are in the same directory.")
    pt, scaler, pca, model_rf = None, None, None, None
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    pt, scaler, pca, model_rf = None, None, None, None

# --- Define the prediction function (CORRECTED) ---
def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                         ph, sulphates, alcohol):
    """
    Predicts the quality of red wine based on input features.
    Applies preprocessing steps (transformation, scaling) before prediction.
    *** Models were trained on SCALED data (11 features), so PCA is NOT applied before predict() ***
    """
    if not all([pt, scaler, pca, model_rf]):
        return "Error: Model or preprocessors failed to load."

    try:
        # 1. Create a DataFrame from the inputs
        input_data = pd.DataFrame([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            ph, sulphates, alcohol
        ]], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ])

        # 2. Apply Power Transformation
        data_transformed = pt.transform(input_data)

        # 3. Apply Scaling
        df_transformed = pd.DataFrame(data_transformed, columns=input_data.columns)
        data_scaled = scaler.transform(df_transformed)

        # 4. Apply PCA - Transform but DON'T use for prediction input here
        # You might still want data_pca if you were displaying components
        data_pca = pca.transform(data_scaled)

        # 5. Make Prediction using SCALED data (11 features)
        prediction = model_rf.predict(data_scaled) # Use data_scaled (11 features)

        predicted_quality = prediction[0]

        return f"Predicted Wine Quality: {predicted_quality}"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Prediction Error: {str(e)}. Check input values."


# --- Define Gradio Input Components ---
input_components = [
    gr.Number(label="Fixed Acidity (g/dm³)", info="e.g., 4.6 - 15.9"),
    gr.Number(label="Volatile Acidity (g/dm³)", info="Tartaric acid, e.g., 0.12 - 1.58"),
    gr.Number(label="Citric Acid (g/dm³)", info="Adds freshness, e.g., 0.0 - 1.0"),
    gr.Number(label="Residual Sugar (g/dm³)", info="Sweetness, e.g., 0.9 - 15.5"),
    gr.Number(label="Chlorides (g/dm³)", info="Saltiness (sodium chloride), e.g., 0.012 - 0.611"),
    gr.Number(label="Free Sulfur Dioxide (mg/dm³)", info="SO2 preservative (free form), e.g., 1 - 72"),
    gr.Number(label="Total Sulfur Dioxide (mg/dm³)", info="SO2 preservative (total), e.g., 6 - 289"),
    gr.Number(label="Density (g/cm³)", info="Density of wine, e.g., 0.99007 - 1.00369"),
    gr.Number(label="pH", info="Acidity/basicity level, e.g., 2.74 - 4.01"),
    gr.Number(label="Sulphates (g/dm³)", info="Potassium sulphate, e.g., 0.33 - 2.0"),
    gr.Number(label="Alcohol (% vol)", info="Alcohol percentage, e.g., 8.4 - 14.9")
]

# --- Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_wine_quality,
    inputs=input_components,
    outputs=gr.Label(label="Prediction Result"),
    title="Red Wine Quality Predictor 🍷",
    description="Enter the chemical properties of a red wine to predict its quality score (typically between 3 and 8). Uses pre-trained preprocessors (PowerTransform, Scaler, PCA) and a Random Forest model.",
    allow_flagging='never',
    examples=[
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
        [11.2, 0.28, 0.56, 1.9, 0.075, 17.0, 60.0, 0.9980, 3.16, 0.58, 9.8],
        [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
    ]
)

if __name__ == "__main__":
    iface.launch()