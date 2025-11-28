import numpy as np
import joblib
import os

MODEL_DIR = "./models"

from tflite_runtime.interpreter import Interpreter


# --------------------------
# LOAD MODELS ONLY ONCE
# --------------------------
interpreter_diet = None
interpreter_int = None
scaler_X = None
heart_model = None   # este será un modelo sklearn ya listo


def load_models_safely():
    global interpreter_diet, interpreter_int, scaler_X, heart_model

    if interpreter_diet is not None:
        return  # Ya cargados

    # --- LOAD ANFIS DIET MODEL ---
    interpreter_diet = Interpreter(
        model_path=os.path.join(MODEL_DIR, "anfis_diet.tflite")
    )
    interpreter_diet.allocate_tensors()

    # --- LOAD ANFIS INTENSITY MODEL ---
    interpreter_int = Interpreter(
        model_path=os.path.join(MODEL_DIR, "anfis_int.tflite")
    )
    interpreter_int.allocate_tensors()

    # --- LOAD SCALER ---
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.joblib"))

    # --- LOAD HEART MODEL ---
    heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_cols.joblib"))


# --------------------------
# MAIN RECOMMENDER
# --------------------------
def recommend_full(data):

    gender = data["gender"]
    age = float(data["age"])
    h = float(data["height_cm"])
    w = float(data["weight_kg"])
    activity = float(data["activity_0_4"])
    goal = data["goal_str"]

    # BMI
    bmi = w / ((h / 100) ** 2)

    # BMR
    if gender == "M":
        bmr = 10 * w + 6.25 * h - 5 * age + 5
    else:
        bmr = 10 * w + 6.25 * h - 5 * age - 161

    tdee = bmr * (1.2 + activity * 0.175)

    # INPUT VECTOR
    X = np.array([[age, h, w, activity, bmi]])

    # SCALE
    Xs = scaler_X.transform(X)

    # ---- HEART RISK ----
    heart_pred_proba = heart_model.predict_proba(Xs)[0, 1]
    heart_bucket = "bajo"
    if heart_pred_proba > 0.66:
        heart_bucket = "alto"
    elif heart_pred_proba > 0.33:
        heart_bucket = "medio"

    # ----------------------------
    # ANFIS DIET MODEL PREDICTION
    # ----------------------------
    diet_input_index = interpreter_diet.get_input_details()[0]["index"]
    diet_output_index = interpreter_diet.get_output_details()[0]["index"]

    interpreter_diet.set_tensor(diet_input_index, Xs.astype(np.float32))
    interpreter_diet.invoke()
    diet_value = interpreter_diet.get_tensor(diet_output_index)[0][0]

    if goal == "fat_burn":
        cal_target = tdee - 350
        diet_type = "déficit"
    elif goal == "muscle_gain":
        cal_target = tdee + 350
        diet_type = "superávit"
    else:
        cal_target = tdee
        diet_type = "mantenimiento"

    # ----------------------------
    # ANFIS INTENSITY MODEL
    # ----------------------------
    int_input_index = interpreter_int.get_input_details()[0]["index"]
    int_output_index = interpreter_int.get_output_details()[0]["index"]

    interpreter_int.set_tensor(int_input_index, Xs.astype(np.float32))
    interpreter_int.invoke()
    int_value = interpreter_int.get_tensor(int_output_index)[0][0]

    if int_value < 0.33:
        int_level = "Baja"
        int_note = "Ejercicio ligero recomendado."
    elif int_value < 0.66:
        int_level = "Media"
        int_note = "Entrenamiento moderado ideal."
    else:
        int_level = "Alta"
        int_note = "Rutina intensa sugerida."

    return {
        "diet": {
            "type": diet_type,
            "calorie_target_kcal": round(cal_target),
            "score": float(diet_value),
        },
        "exercise": {
            "intensity": int_level,
            "note": int_note,
            "score": float(int_value),
        },
        "heart_risk": {
            "bucket": heart_bucket,
            "prob": round(float(heart_pred_proba), 4),
        },
        "body": {
            "bmi": round(bmi, 2),
            "bmr": round(bmr),
            "tdee": round(tdee),
        },
    }
