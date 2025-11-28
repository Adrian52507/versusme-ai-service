import numpy as np
import joblib
import os
from tflite_runtime.interpreter import Interpreter

MODEL_DIR = "/app/models"   # ruta correcta para Railway

# VARIABLES GLOBALES
interpreter_diet = None
interpreter_int = None
scaler_X = None
heart_model = None


# ============================================================
# LOAD MODELS ONCE (CORRECTO)
# ============================================================
def load_models_safely():
    global interpreter_diet, interpreter_int, scaler_X, heart_model

    # Si ya están cargados → no recargar
    if interpreter_diet is not None:
        return

    print("📌 Cargando modelos de:", MODEL_DIR)

    # Listar archivos para debug
    try:
        print("📂 Archivos en /app/models:", os.listdir(MODEL_DIR))
    except Exception as e:
        print("❌ ERROR listando modelos:", e)

    # ANFIS DIET
    interpreter_diet = Interpreter(
        model_path=os.path.join(MODEL_DIR, "anfis_diet.tflite")
    )
    interpreter_diet.allocate_tensors()

    # ANFIS INTENSITY
    interpreter_int = Interpreter(
        model_path=os.path.join(MODEL_DIR, "anfis_int.tflite")
    )
    interpreter_int.allocate_tensors()

    # SCALER
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.joblib"))

    # HEART ML MODEL
    heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_clf.joblib"))

    print("✅ Modelos cargados correctamente")


# ============================================================
# MAIN RECOMMENDER
# ============================================================
def recommend_full(data):

    # Asegurar que los modelos están cargados
    load_models_safely()

    gender = data["gender"]
    age = float(data["age"])
    h = float(data["height_cm"])
    w = float(data["weight_kg"])
    activity = float(data["activity_0_4"])
    goal = data["goal_str"]

    # --- BODY METRICS ---
    bmi = w / ((h / 100) ** 2)

    if gender == "M":
        bmr = 10 * w + 6.25 * h - 5 * age + 5
    else:
        bmr = 10 * w + 6.25 * h - 5 * age - 161

    tdee = bmr * (1.2 + activity * 0.175)

    X = np.array([[age, h, w, activity, bmi]])

    # SCALE
    Xs = scaler_X.transform(X)

    # --- HEART RISK ---
    heart_pred_proba = heart_model.predict_proba(Xs)[0, 1]
    if heart_pred_proba > 0.66:
        heart_bucket = "alto"
    elif heart_pred_proba > 0.33:
        heart_bucket = "medio"
    else:
        heart_bucket = "bajo"

    # --- ANFIS DIET ---
    diet_in = interpreter_diet.get_input_details()[0]["index"]
    diet_out = interpreter_diet.get_output_details()[0]["index"]
    interpreter_diet.set_tensor(diet_in, Xs.astype(np.float32))
    interpreter_diet.invoke()
    diet_value = interpreter_diet.get_tensor(diet_out)[0][0]

    if goal == "fat_burn":
        cal_target = tdee - 350
        diet_type = "déficit"
    elif goal == "muscle_gain":
        cal_target = tdee + 350
        diet_type = "superávit"
    else:
        cal_target = tdee
        diet_type = "mantenimiento"

    # --- ANFIS INTENSITY ---
    int_in = interpreter_int.get_input_details()[0]["index"]
    int_out = interpreter_int.get_output_details()[0]["index"]
    interpreter_int.set_tensor(int_in, Xs.astype(np.float32))
    interpreter_int.invoke()
    int_value = interpreter_int.get_tensor(int_out)[0][0]

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
