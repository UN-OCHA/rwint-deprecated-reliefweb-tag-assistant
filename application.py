# Initializing the model
import socket

from flask import Flask, request
from flask import make_response
from flask_cors import CORS, cross_origin

from reliefweb_tag import reliefweb_ml_model, reliefweb_predict, reliefweb_config

global application
application = Flask(__name__)
cors = CORS(application)
application.config['CORS_HEADER'] = 'Content-type'
# Content-type: application/json
application.debug = False
application.threaded = False

global models
models = {}


def download_nltk_corpus():
    # from https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py
    # -*- coding: utf-8 -*-
    # Downloads the necessary NLTK models and corpora required to support
    # all of newspaper's features. Modify for your own needs.
    import nltk
    import os

    # this is only needed to build the models from scratch
    required_nltk_corpora = [
        'brown',  # Required for FastNPExtractor
        'punkt',  # Required for WordTokenizer
        'maxent_treebank_pos_tagger',  # Required for NLTKTagger
        'movie_reviews',  # Required for NaiveBayesAnalyzer
        'wordnet',  # Required for lemmatization and Wordnet
        'stopwords'
    ]
    try:
        downloaded_nltk_corpora = [os.listdir(nltk.data.find("corpora")),
                                   os.listdir(nltk.data.find("tokenizers")),
                                   os.listdir(nltk.data.find("taggers"))]
    except Exception as e:
        print("Empty nltk corpus, initialiting")
        downloaded_nltk_corpora = []

    print("Start downloading nltk corpora.", flush=True)
    for each in required_nltk_corpora:
        if each not in downloaded_nltk_corpora:
            print(('Downloading "{0}"'.format(each)))
            nltk.download(each, download_dir=reliefweb_config.NLTK_DATA_PATH)
    print("Finished downloading nltk corpora.")
    return

download_nltk_corpus()

print("** In the main flow **")

def init_models():
    print("Testing if machine learning models exist")
    for each in reliefweb_config.MODEL_NAMES:
        # TODO: What does this language collector means? Can we remove it?
        if models.get(each, '') == '':
            print("> MAIN: Creating neural network for " + each)
            model = reliefweb_ml_model.ReliefwebModel(each)
            models[each] = model


# Creating the API endpoints
@application.route("/")
# Instructions ENDPOINT
@cross_origin()
def main():
    return "Please, use the /tag_url endpoint with the param url to tag a url or pdf. Example: http://IP:PORT/tag_url?url=URL_WITH_HTTP"


@application.route("/tag_url")
# sample http://localhost:5000/tag_url?scope=report&url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
@cross_origin()
def reliefweb_tag_url():

    import gc
    import json

    gc.collect()
    url = request.args.get('url')
    scope = request.args.get('scope')
    # if (RWModel.get('language', '') == '') or (RWModel.get('theme', '') == ''):
    sample_dict = reliefweb_predict.process_url_input(url)  # metadata and scraping the article
    init_models()
    if scope in ["report", "job"]:
        sample_dict = reliefweb_predict.predict(_models=models, _sample_dict=sample_dict, _scope=scope)
        # machine learning predictions
    else:
        sample_dict = {"error": "scope parameter should be job or report", "full_text": ""}
    print("\nDone prediction for: " + url)

    response = make_response(json.dumps(sample_dict, indent=4))
    response.headers['content-type'] = 'application/json'
    return response


@application.route("/tag_text", methods=['POST', 'GET'])
# GET sample http://localhost:5000/tag_text?scope=job&text=Blablalblbalblalbalñldfjk
@cross_origin()
def reliefweb_tag_text():
    import gc
    import json

    gc.collect()

    if request.method == 'POST':  # the request will be always POST from the HTML frontend
        text = request.form['text']
        scope = request.form['scope']
    else:  # The get has limitations in the size of the text as it is on the URL
        text = request.args.get('text')
        scope = request.args.get('scope')

    sample_dict = reliefweb_predict.process_text_input(_input=text)
    init_models()
    if scope in ["report", "job"]:
        sample_dict = reliefweb_predict.predict(_models=models, _sample_dict=sample_dict, _scope=scope)
    else:
        sample_dict = {"error": "scope parameter should be job or report", "full_text": ""}
    print("\nDone prediction for: " + str(text)[:20] + "...")

    response = make_response(json.dumps(sample_dict, indent=4))
    response.headers['content-type'] = 'application/json'
    return response

@application.route("/html")
@cross_origin()
def htmlpage():
    return app.send_static_file('rwtag.html')


@application.route("/test")
def test():
    return "TEST ENDPOINT"


if __name__ == '__main__':
    # get public IP -- if needed
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    publicIP = s.getsockname()[0]
    s.close()

    # application.run(debug=reliefweb_config.DEBUG, host=publicIP, port=reliefweb_config.PORT)  # use_reloader=False
    init_models()
    application.run(debug=reliefweb_config.DEBUG, host='0.0.0.0', port=reliefweb_config.PORT)  # use_reloader=False // This does not call to main
