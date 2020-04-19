import os
from flask import Flask, flash, redirect, render_template, request, session, abort
from lstm import LSTMModel
from clear_bash import clear_bash

app = Flask(__name__)
cleaner=clear_bash()

def getModel() :
    global lstmModel
    lstmModel = LSTMModel()

@app.route("/")
def index():

    return render_template('index.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form.getlist('Job')
      getModel()
      processed_text = lstmModel.predictCategory(result[0])
      result = {'Job': processed_text}
      return render_template("result.html",result = result)

def clear_bash():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    clear_bash()
    print("---------------------------------")
    print("NEWS CLASSIFICATION APP")
    print("---------------------------------")
    
    clear_bash()
    print(("*Flask starting server..."
                   "please wait until server has fully started"))
    app.run()

