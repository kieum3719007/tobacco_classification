import io
import os

from flask import Flask, render_template, request
from PIL import Image

from models.classifiers import get_classifier, get_labels, load
from models.feature_extractors import BackboneName
from transformations.transformations import (TRANSFORMATIONS,
                                             get_transformations)

app = Flask(__name__)
labels = get_labels()
model = get_classifier(BackboneName.VGG19, len(labels))
model = load(model)


def transform_image(image_bytes):

    transform = get_transformations(TRANSFORMATIONS)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # read image bytes
        file = request.files['image']
        if (file):
            image_bytes = file.read()
        
            save_image(image_bytes)
            confidence, class_name = get_prediction(image_bytes)
            return render_template("result.html", confidence=confidence, class_name=class_name, bytes=image_bytes)
        else:
            return render_template('index.html')


def get_prediction(image_bytes):
    """
    Gets the prediction of a single image
    """
    image = transform_image(image_bytes)
    outputs = model.forward(image)
    confidence, index = outputs.max(1)
    confidence, label = confidence.item(), labels[index.item()]
    return confidence, label


def save_image(image_bytes):
    """
    Saves an image to temporary location
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((500, 500))

    image.save(os.path.join("static", "images", "temp.png"))
