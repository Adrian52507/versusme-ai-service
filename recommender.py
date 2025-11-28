import os
import numpy as np
import joblib
import pandas as pd
from tflite_runtime.interpreter import Interpreter

# Ruta a los modelos
model_dir = os.path.dirname(__file__)

# Variables globales (inicializadas en load_models_safely)
interpreter_int = None
interpreter_diet = None
scaler_X = None
clf = None
heart_cols = None


# ------------------------------------------
# CARGA SEGURA DE MODELOS (Railway compatible)
# ------------------------------------------
def load_models_safely():
    global interpreter_int, interpreter_diet, scaler_X, clf, heart_cols

    if interpreter_int is None:
        interpreter_int = Interpreter(model_path=os.path.join(model_dir, "anfis_int.tflite"))
        interpreter_int.allocate_tensors()

    if interpreter_diet is None:
        interpreter_diet = Interpreter(model_path=os.path.join(model_dir, "anfis_diet.tflite"))
        interpreter_diet.allocate_tensors()

    if scaler_X is None:
        scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.joblib"))

    if clf is None:
        clf = joblib.load(os.path.join(model_dir, "heart_clf.joblib"))

    if heart_cols is None:
        try:
            heart_cols = joblib.load(os.path.join(model_dir, "heart_cols.joblib"))
        except:
            heart_cols = None


# -------------------------
# ANFIS PREDICTOR
# -------------------------
def anfis_predict(interpreter, x):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0, 0]


# -------------------------
# FUNCIÓN PRINCIPAL IA
# -------------------------
def recommend_full(gender_str, age, height_cm, weight_kg, activity_0_4, goal_str):
    # Cálculos básicos cuerpo
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender_str.lower()[0] == 'm' else -161)
    PAL = [1.2, 1.375, 1.55, 1.725, 1.9][activity_0_4]
    tdee = bmr * PAL

    goal_map = {'fat_burn': -400, 'maintain': 0, 'muscle_gain': 400}
    kcal = tdee + goal_map.get(goal_str, 0)

    # Vector ANFIS
    gender_num = 0 if gender_str.lower()[0] == 'm' else 1
    goal_num = {'fat_burn': 0, 'maintain': 1, 'muscle_gain': 2}.get(goal_str, 1)

    x_raw = np.array([[gender_num, age, height_cm, weight_kg, activity_0_4, goal_num, bmi]])
    x_cont = scaler_X.transform(x_raw[:, [1, 2, 3, 6]])
    x_cat = x_raw[:, [0, 4, 5]]
    x_in = np.concatenate([x_cont, x_cat], axis=1).astype(np.float32)

    # Predicciones
    c_int = int(np.clip(np.rint(anfis_predict(interpreter_int, x_in)), 0, 2))
    c_diet = int(np.clip(np.rint(anfis_predict(interpreter_diet, x_in)), 0, 2))

    INT2TXT = ['Ligero', 'Moderado', 'Intenso']
    DIET2TXT = ['Hipocalórica', 'Balanceada', 'Proteica']

    # Riesgo cardiaco
    df = build_heart_features(age, gender_str, bmi, activity_0_4)
    p_hd = float(clf.predict_proba(df)[0, 1])
    risk = 'Bajo' if p_hd < 0.12 else 'Medio' if p_hd < 0.25 else 'Alto'

    return {
        "exercise": {
            "intensity": INT2TXT[c_int],
            "note": intensity_note(c_int),
        },
        "diet": {
            "type": DIET2TXT[c_diet],
            "calorie_target_kcal": int(round(kcal)),
        },
        "heart_risk": {
            "prob": round(p_hd, 3),
            "bucket": risk,
        },
        "body": {
            "bmi": round(bmi, 2),
            "bmr": round(bmr, 1),
            "tdee": round(tdee, 1),
        }
    }


# -------------------------
# HELPERS (necesarios)
# -------------------------
def intensity_note(level):
    if level == 0:
        return "Inicia suave: 2–3 sesiones/semana."
    if level == 1:
        return "Mantén 3–4 sesiones/semana."
    return "Planifica 4–5 sesiones/semana plus intervalos."


def build_heart_features(age, gender, bmi, activity, cols=None):
    sex = 'Male' if gender.lower().startswith('m') else 'Female'
    df = pd.DataFrame([{
        "BMI": bmi,
        "Sex": sex,
        "AgeCategory": "18-24" if age < 25 else "25-29",
        "PhysicalActivity": "Yes" if activity > 0 else "No",
    }])
    return df
