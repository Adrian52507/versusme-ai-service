import os
import numpy as np
import joblib
import pandas as pd
from tflite_runtime.interpreter import Interpreter

model_dir = os.path.dirname(__file__)

# Variables globales
interpreter_int = None
interpreter_diet = None
scaler_X = None
clf = None
heart_cols = None

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
