from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Global variables to store dataset & model
df = None
model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global df, model
    try:
        file = request.files.get("dataset")
        if not file:
            return render_template("index.html", error="❌ No file uploaded")

        # Read dataset directly
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Clean dataset
        df.columns = df.columns.str.strip()
        if "Date" not in df.columns or "Energy" not in df.columns:
            return render_template("index.html",
                                   error="❌ Dataset must contain 'Date' and 'Energy' columns.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Add Day_Index
        df["Day_Index"] = range(len(df))

        # Train model once here
        X = df[["Day_Index"]]
        y = df["Energy"]
        model = LinearRegression()
        model.fit(X, y)

        return render_template("index.html",
                               message="Dataset uploaded & model trained successfully!",
                               latest_date=df["Date"].max().date(),
                               latest_energy=df["Energy"].iloc[-1])
    except Exception as e:
        return render_template("index.html", error=f"❌ Error: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    global df, model
    try:
        if df is None or model is None:
            return render_template("index.html", error="❌ Please upload dataset first!")

        # Predict next day
        next_day_index = len(df)
        next_day_prediction = model.predict([[next_day_index]])[0]
        next_date = df["Date"].max() + pd.Timedelta(days=1)

        result = f"🔮 Predicted Energy Consumption for {next_date.date()}: {next_day_prediction:.2f} kWh"

        return render_template("index.html",
                               result=result,
                               latest_date=df["Date"].max().date(),
                               latest_energy=df["Energy"].iloc[-1])
    except Exception as e:
        return render_template("index.html", error=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
