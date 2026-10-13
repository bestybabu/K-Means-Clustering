readme_content = """# Breast Cancer Co-relation Predictor App

This project implements a machine learning system designed to predict breast cancer diagnostic status (Healthy Control vs. Patient) using metabolic and clinical indicators. It utilizes a hybrid model structure incorporating K-Means clustering assignments as engineered features alongside standard biometric inputs for classification models.

## Project Directory Contents
- `datar2.csv`: The raw clinical dataset.
- `model_artifacts.pkl`: Saved pre-processing pipeline (StandardScaler), trained K-Means model (k=2), and the best performing classifier (K-Nearest Neighbors).
- `app.py`: Streamlit-based web application to input patient data and generate interactive predictions.
- `readme.txt`: This documentation file.

## Clinical Features Used
1. Age (years)
2. BMI (kg/m²)
3. Glucose (mg/dL)
4. Insulin (µU/mL)
5. HOMA (Homeostatic Model Assessment of insulin resistance)
6. Leptin (ng/mL)
7. Adiponectin (µg/mL)
8. Resistin (ng/mL)
9. MCP-1 (Monocyte Chemoattractant Protein-1, pg/mL)

## How to Run the App Locally

1. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Navigate to the local URL (usually `http://localhost:8501`) shown in your terminal.
"""

with open('readme.txt', 'w') as f:
    f.write(readme_content)

print("readme.txt file has been created successfully!")
