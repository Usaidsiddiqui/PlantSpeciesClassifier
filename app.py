from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import joblib
from  gnbmodel import *
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_name = request.form.get("model")
        print(model_name)
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("i am jsc ch cwe chwec wc  cbwe c")
            print(model_name)

            # Call the main_function from your model.py
            result = main_function(filepath, model_name)
            
            return render_template("result.html", result=result)
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)
