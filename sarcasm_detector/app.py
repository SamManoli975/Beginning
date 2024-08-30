from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('sarcasm_detector_model')

app= Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        input_text = request.form['text_input']

        
