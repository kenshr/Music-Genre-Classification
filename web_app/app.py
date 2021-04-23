from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load( open( "../deploy/nmf_model.pkl", "rb" ) )

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html', current_page='HOME')

@app.route('/about', methods=['GET'])
def about():
  return render_template('about.html' ,current_page='ABOUT')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8105, debug=True)