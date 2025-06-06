import pandas as pd
import joblib
import requests
from io import BytesIO
import gdown

# Google Drive에서 모델 파일 다운로드
file_id = "1ODV76nTt25sles3hwNVJ6dMcsu0JSTVE"
output_path = "smoking_model.pkl"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# 모델 불러오기
model = joblib.load(output_path)

# Input widgets with units
sex_widget = widgets.RadioButtons(options=['Male', 'Female'], description='Sex:')
age_widget = widgets.IntSlider(value=40, min=10, max=100, step=1, description='Age (years):')
height_widget = widgets.FloatSlider(value=170, min=130, max=210, step=0.5, description='Height (cm):')
weight_widget = widgets.FloatSlider(value=70, min=30, max=150, step=0.5, description='Weight (kg):')
waistline_widget = widgets.FloatSlider(value=85, min=50, max=150, step=0.5, description='Waistline (cm):')
sbp_widget = widgets.IntSlider(value=120, min=80, max=200, step=1, description='SBP (mmHg):')
dbp_widget = widgets.IntSlider(value=80, min=40, max=140, step=1, description='DBP (mmHg):')
blds_widget = widgets.FloatSlider(value=90, min=50, max=300, step=1, description='Blood Sugar (mg/dL):')
tot_chole_widget = widgets.FloatSlider(value=180, min=100, max=400, step=1, description='Total Cholesterol (mg/dL):')
hdl_widget = widgets.FloatSlider(value=50, min=20, max=100, step=1, description='HDL (mg/dL):')
ldl_widget = widgets.FloatSlider(value=100, min=50, max=300, step=1, description='LDL (mg/dL):')
tg_widget = widgets.FloatSlider(value=150, min=50, max=500, step=1, description='Triglyceride (mg/dL):')
ggtp_widget = widgets.FloatSlider(value=30, min=10, max=300, step=1, description='Gamma-GTP (U/L):')

# Button and output
predict_button = widgets.Button(description="Predict Smoking Status")
output = widgets.Output()

# Prediction function
def predict_smoking_status(b):
    with output:
        clear_output()

        user_data = {
            'age': age_widget.value,
            'height': height_widget.value,
            'weight': weight_widget.value,
            'waistline': waistline_widget.value,
            'SBP': sbp_widget.value,
            'DBP': dbp_widget.value,
            'BLDS': blds_widget.value,
            'tot_chole': tot_chole_widget.value,
            'HDL_chole': hdl_widget.value,
            'LDL_chole': ldl_widget.value,
            'triglyceride': tg_widget.value,
            'gamma_GTP': ggtp_widget.value,
            'sex_Female': 1 if sex_widget.value == 'Female' else 0,
            'sex_Male': 1 if sex_widget.value == 'Male' else 0
        }

        user_df = pd.DataFrame([user_data])
        user_df = user_df.reindex(columns=model.feature_names_in_, fill_value=0)

        print("\nInput values used for prediction:")
        display(user_df)

        prediction = model.predict(user_df)[0]
        prob = model.predict_proba(user_df)[0][prediction]

        if prediction == 1:
            print(f"Prediction: Smoker (Probability: {prob:.2f})")
        else:
            print(f"Prediction: Non-Smoker (Probability: {prob:.2f})")

# Bind prediction function
predict_button.on_click(predict_smoking_status)

# Display UI
display(widgets.VBox([
    sex_widget, age_widget, height_widget, weight_widget, waistline_widget,
    sbp_widget, dbp_widget, blds_widget, tot_chole_widget, hdl_widget,
    ldl_widget, tg_widget, ggtp_widget,
    predict_button, output
]))
