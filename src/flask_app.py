from flask import Flask, render_template, request, jsonify
from detector import Detector
from keras.models import load_model
from helpers import predict_images
from solver import parse_and_solve
import os
import cv2

app = Flask(__name__, template_folder='templates')


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("home.html", task='0')


@app.route("/upload", methods=["POST", "GET"])
def upload_picture():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save("image.png")
            detector = Detector(verbose=False)
            crop_list, crop_coord = detector.detect("image.png")
            model = load_model("../model")

            task = predict_images(model, crop_list, verbose=False)
            os.remove("image.png")
            print(task)
            solution = parse_and_solve(task)
            if solution is not None:
                print("Uspješno rješavanje izraza: ", task)
                print("Rješenje je: ", solution)
            else:
                print("Pročitani izraz glasi: ", task)
                print("Nažalost, izraz je pogrešno pročitan ili pogrešno zadan.")
            data = {'solution': solution, 'task': task}
            return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
