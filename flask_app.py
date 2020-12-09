from flask import Flask, render_template, request, jsonify
from src.detector import Detector
from keras.models import load_model
from src.helpers import predict_images
from src.solver import parse_and_solve
import os
import cv2
import time

app = Flask(__name__, template_folder='src/templates')


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("home.html", task='0')


@app.route("/upload", methods=["POST", "GET"])
def upload_picture():
    if request.method == "POST":
        if request.files:
            time.sleep(3)
            image = request.files["image"]
            image.save("image.png")
            detector = Detector(verbose=False)
            crop_list, crop_coord = detector.detect("image.png")
            model = load_model(os.path.dirname(os.path.abspath(__file__)) + "/model")

            task = predict_images(model, crop_list, verbose=False)
            os.remove("image.png")
            print(task)
            solution = parse_and_solve(task)
            data = {'solution': solution, 'task': task}
            return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
