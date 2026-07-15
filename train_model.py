from pathlib import Path
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

with open(BASE_DIR / "model.pkl", "rb") as file:
    model = pickle.load(file)

CROP_DETAILS = {
    "rice": ("Rice", "High rainfall and humid conditions are suitable."),
    "maize": ("Maize", "Moderate temperature and well-drained soil are preferred."),
    "chickpea": ("Chickpea", "Cool and relatively dry conditions are suitable."),
    "kidneybeans": ("Kidney Beans", "Mild climate with moderate rainfall is suitable."),
    "pigeonpeas": ("Pigeon Peas", "Warm weather and medium rainfall are suitable."),
    "mothbeans": ("Moth Beans", "Hot and dry conditions are suitable."),
    "mungbean": ("Mung Bean", "Warm and humid conditions with light rainfall are suitable."),
    "blackgram": ("Black Gram", "Warm climate with moderate humidity is suitable."),
    "lentil": ("Lentil", "Cool and dry growing conditions are suitable."),
    "pomegranate": ("Pomegranate", "Warm conditions with high humidity are suitable."),
    "banana": ("Banana", "Warm and humid tropical conditions are suitable."),
    "mango": ("Mango", "Hot climate with moderate humidity is suitable."),
    "grapes": ("Grapes", "Mild temperature and potassium-rich soil are preferred."),
    "watermelon": ("Watermelon", "Warm weather with high humidity is suitable."),
    "muskmelon": ("Muskmelon", "Hot and humid conditions with low rainfall are suitable."),
    "apple": ("Apple", "Cool temperature and humid conditions are suitable."),
    "orange": ("Orange", "Mild climate and slightly acidic soil are suitable."),
    "papaya": ("Papaya", "Hot and humid conditions with good rainfall are suitable."),
    "coconut": ("Coconut", "Tropical climate with very high humidity is suitable."),
    "cotton": ("Cotton", "Warm conditions and moderate rainfall are suitable."),
    "jute": ("Jute", "Warm and humid climate with high rainfall is suitable."),
    "coffee": ("Coffee", "Mild temperature and high rainfall are suitable.")
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    confidence = None
    note = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            values = {feature: float(request.form[feature]) for feature in FEATURES}

            if not (0 <= values["N"] <= 300 and 0 <= values["P"] <= 300 and 0 <= values["K"] <= 350):
                raise ValueError("N, P and K must be within the supported range.")
            if not (0 <= values["temperature"] <= 60):
                raise ValueError("Temperature must be between 0 and 60°C.")
            if not (0 <= values["humidity"] <= 100):
                raise ValueError("Humidity must be between 0 and 100%.")
            if not (0 <= values["ph"] <= 14):
                raise ValueError("pH must be between 0 and 14.")
            if not (0 <= values["rainfall"] <= 500):
                raise ValueError("Rainfall must be between 0 and 500 mm.")

            input_data = pd.DataFrame(
                [[values[name] for name in FEATURES]],
                columns=FEATURES
            )

            predicted_key = model.predict(input_data)[0]
            result, note = CROP_DETAILS.get(
                predicted_key,
                (predicted_key.title(), "Suitable for the supplied conditions.")
            )

            if hasattr(model, "predict_proba"):
                confidence = round(float(model.predict_proba(input_data).max()) * 100, 1)

        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        result=result,
        confidence=confidence,
        note=note,
        error=error,
        values=values
    )

if __name__ == "__main__":
    app.run(debug=True)
