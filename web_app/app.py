from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# model = pickle.load(open( "../model/", "rb" ) )

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html', current_page='INDEX')

@app.route('/about', methods=['GET'])
def about():
  return render_template('about.html' ,current_page='ABOUT')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8105, debug=True)