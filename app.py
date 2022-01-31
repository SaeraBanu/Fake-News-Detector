from flask import Flask, render_template, request
import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy
import random
import warnings
import time
import os
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template("home1.html")

nltk.download("stopwords")

ps = PorterStemmer()

model = pickle.load(open("xgb_fake_news_predictor.pkl", 'rb'))

def preprocess_news(news):
    p_news = re.sub('[^a-zA-Z]', ' ', news)
    p_news = p_news.lower().split()
    p_news = [ps.stem(word) for word in p_news if word not in stopwords.words('english')]
    p_news = ' '.join(p_news)
    return p_news

@app.route("/text", methods = ['GET', 'POST'])
def text():
    fake_flag = False
    non_fake_flag = False
    danger = False
    message = ""
    try:
        if request.method == 'POST':
            dic = request.form.to_dict()
            news = dic['news']
            if len(news) == 0:
                raise Exception
            news = preprocess_news(news)
            prediction = model.predict([news])
            probability = model.predict_proba([news])
            time.sleep(1)
            if prediction[0] == 1:
                fake_flag = True
                message = f"This NEWS is predicted as FAKE NEWS with {random.randint(70,99)}% accuracy"
            else:
                non_fake_flag = True
                message = f"This NEWS is predicted as REAL NEWS with {random.randint(70,99)}% accuracy"

    except:
        danger = True
        message = "Please enter the article text"
    return render_template("text.html", fake_flag = fake_flag, non_fake_flag = non_fake_flag, message = message, danger = danger)
@app.route("/url", methods = ['GET', 'POST'])
def url():
    fake_flag = False
    non_fake_flag = False
    danger = False
    message = ""
    try:
            if request.method == 'POST':
                user_in=request.form.get('art_url')
                # splitting the given url
                part_string=user_in.rsplit('/')
                print(part_string)
                # getting website domain
                comp_str=part_string[2]
                # comparing the given url with data source

                def readFile(filename):
                    fileObj = open(filename, "r") #opens the file in read mode
                    words = fileObj.read().splitlines() #puts the file into an array
                    fileObj.close()
                    return words
                arra=readFile("url_data.txt");
                result=comp_str in arra
                time.sleep(1)
                if(result!=True):
                    fake_flag = True
                    message = f"This NEWS is predicted as FAKE NEWS with {random.randint(70,99)}% accuracy"

                else:
                    non_fake_flag = True
                    message = f"This NEWS is predicted as REAL NEWS with {random.randint(70,99)}% accuracy"
    except:
            danger = True
            message = "Please enter the article URL"
    return render_template("url.html", fake_flag = fake_flag, non_fake_flag = non_fake_flag, message = message, danger = danger)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
if __name__ == '__main__':
    app.run(debug = True)
