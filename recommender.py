import numpy as np
import os
import joblib
import pandas as pd

# Intentar usar tflite_runtime; si falla, usar tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime")
except ImportError:
    from tensorflow.lite import Interpreter
    print("Using tensorflow.lite")

MODEL_DIR = "./models"

interpreter_cal = None
interpreter_int = None


# 🔹 Cargar modelo TFLite ANFIS de forma segura
def load_models_safely():
    global interpreter_cal, interpreter_int

    if interpreter_cal is None:
        interpreter_cal = Interpreter(
            model_path=os.path.join(MODEL_DIR, "anfis_cal.tflite")
        )
        interpreter_cal.allocate_tensors()

    if interpreter_int is None:
        interpreter_int = Interpreter(
            model_path=os.path.join(MODEL_DIR, "anfis_int.tflite")
        )
        interpreter_int.allocate_tensors()


# 🔹 Calcular BMR (Harris-Benedict)
def compute_bmr(gender, weight, height, age):
    if gender == "M":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)


# 🔹 Recomendación completa (ejercicio + dieta + calorías)
def recommend_full(payload):
    load_models_safely()

    gender = payload["gender"]
    age = payload["age"]
    height = payload["height_cm"]
    weight = payload["weight_kg"]
    activity = payload["activity_0_4"]
    goal = payload["goal_str"]

    # ⭐ BMI
    bmi = weight / ((height / 100) ** 2)

    # ⭐ BMR
    bmr = compute_bmr(gender, weight, height, age)

    # ⭐ TDEE base (factor actividad estándar)
    tdee = bmr * (1.2 + 0.15 * activity)

    # ----------- ANFIS para calorías -----------
    cal_input = np.array([[age, height, weight, activity]], dtype=np.float32)

    cal_i = interpreter_cal.get_input_details()[0]
    cal_o = interpreter_cal.get_output_details()[0]

    interpreter_cal.set_tensor(cal_i["index"], cal_input)
    interpreter_cal.invoke()
    cal_pred = float(interpreter_cal.get_tensor(cal_o["index"])[0][0])

    if goal == "fat_burn":
        target_kcal = cal_pred - 350
        diet_type = "Déficit moderado"
    elif goal == "muscle_gain":
        target_kcal = cal_pred + 250
        diet_type = "Superávit controlado"
    else:
        target_kcal = cal_pred
        diet_type = "Mantenimiento"

    # ----------- ANFIS para intensidad ejercicio -----------
    int_input = np.array([[age, weight, activity]], dtype=np.float32)
    int_i = interpreter_int.get_input_details()[0]
    int_o = interpreter_int.get_output_details()[0]

    interpreter_int.set_tensor(int_i["index"], int_input)
    interpreter_int.invoke()
    intensity_pred = float(interpreter_int.get_tensor(int_o["index"])[0][0])

    # Mapear intensidad a palabra
    if intensity_pred < 0.4:
        intensity = "Baja"
        note = "Entrenamientos suaves, ideal para iniciar."
    elif intensity_pred < 0.7:
        intensity = "Media"
        note = "Entrenamientos balanceados 3-4 veces por semana."
    else:
        intensity = "Alta"
        note = "Entrenamientos intensos, apto para usuarios avanzados."

    # ----------- RESPUESTA FINAL (sin heart_risk) -----------
    return {
        "exercise": {
            "intensity": intensity,
            "note": note
        },
        "diet": {
            "type": diet_type,
            "calorie_target_kcal": int(target_kcal)
        },
        "body": {
            "bmi": round(bmi, 2),
            "bmr": int(bmr),
            "tdee": int(tdee)
        }
    }
