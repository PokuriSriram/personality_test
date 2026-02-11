from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("personality_model.pkl", "rb") as file:
    model, columns = pickle.load(file)

@app.route("/", methods=["GET","POST"])
def index():
    result = None

    if request.method == "POST":
        answers = []
        for i in range(10):
            val = int(request.form.get(f"q{i+1}"))
            answers.append(val)

        prediction = model.predict([answers])[0]

        if prediction == 1:
            result = "You are an Extrovert 😎"
        else:
            result = "You are an Introvert 🤫"

    return render_template("index.html", result=result)

app.run(debug=True)
