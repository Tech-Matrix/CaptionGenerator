from flask import Flask, render_template, request
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from werkzeug.utils import secure_filename
from keras.models import load_model
from pickle import load
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, redirect, url_for, request, render_template
from PIL import Image
import sys
import os

tokenizer = load(open('Tokenizer.p', 'rb'))
model = load_model('model_8.h5')

def extract_features(filename, model):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    if x.shape[2] == 4:
        x = x[..., :3]
    x = np.expand_dims(x, axis=0)
    x = x / 127.5
    x = x - 1.0

    preds = model.predict(x)
    return preds


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


app = Flask(__name__)

@app.route('/', methods=['GET'])
def upload():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        max_length = 32
        xception_model = Xception(include_top=False, pooling="avg")
        photo = extract_features(file_path, xception_model)
        des = generate_desc(model, tokenizer, photo, max_length)
        return des
    return None


if __name__ == '__main__':
    app.run(debug=True)