# Initializing the model
import socket

from flask import Flask, request
from flask_cors import CORS, cross_origin

from reliefweb_tag import reliefweb_ml_model, reliefweb_predict, reliefweb_config

global app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER']='Content-type'
#Content-type: application/json
app.debug = False
app.threaded = False

global RWModel
RWModel = {}

print("** In the main flow **")


def init():
    print("Initializing the ReliefWeb Tag Assistant: auto-tag urls using RW Tags and Machine Learning")

    # Creating the API endpoints
    REQUIRED_CORPORA = [
        'brown',  # Required for FastNPExtractor
        'punkt',  # Required for WordTokenizer
        'maxent_treebank_pos_tagger',  # Required for NLTKTagger
        'movie_reviews',  # Required for NaiveBayesAnalyzer
        'wordnet',  # Required for lemmatization and Wordnet
        'stopwords'
    ]

    print("Start downloading nltk corpora.", flush=True)

    import nltk

    for each in REQUIRED_CORPORA:
        print(('Downloading "{0}"'.format(each)))
        nltk.download(each)

    print("Finished downloading nltk corpora.")

    # from https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py
    # -*- coding: utf-8 -*-
    # Downloads the necessary NLTK models and corpora required to support
    # all of newspaper's features. Modify for your own needs.

    print("> Initializing machine learning model")
    # model_language = reliefweb_ml_model.ReliefwebModel()

    # print("> MAIN: Creating neural network for language")
    # as the language model is accurate, we reduce its complexity for gaining on memory
    # model_language.create_train_one_tag_model(vocabulary_name='language',
    #                                  vocabulary_file=reliefweb_config.DATA_PATH +
    #                                                  reliefweb_config.DATASETS["language"]["vocabulary"],
    # path+"rw-languages.csv",
    #                                  dataset_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["language"][
    #                                     "dataset"],
    #                                  # path+'report_language-1k.csv',
    #                                  model_path=reliefweb_config.MODEL_PATH,
    #                                  term_field='language',
    #                                  vocabulary_language='',  # values: English, Spanish, French, ''
    #                                  dataset_post_field='post',
    #                                  dataset_tag_field='language_name',
    #                                  max_words=reliefweb_config.MAX_WORDS,
    #                                  # 20000, number of words to take from each post to tokenize),
    #                                  batch_size=reliefweb_config.BATCH_SIZE_LANG,
    #                                  epochs=reliefweb_config.EPOCHS,
    #                                  train_percentage=reliefweb_config.TRAINING_PERCENTAGE,  # 0.9
    #                                  skip_normalizing=reliefweb_config.FAST_TESTING
    #                                  )
    print("> MAIN: Creating neural network for themes")
    model_theme = reliefweb_ml_model.ReliefwebModel()
    model_theme.create_train_one_tag_model(vocabulary_name='theme',
                                   vocabulary_language='',  # values: English, Spanish, French, ''
                                   vocabulary_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["theme"][
                                       "vocabulary"],  # path+"rw-themes.csv",
                                   dataset_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["theme"][
                                       "dataset"],
                                   # path+'report_theme_uneven_multiple-30k.csv',#'report_theme_en-1k.csv',#
                                   model_path=reliefweb_config.MODEL_PATH,
                                   term_field='theme',
                                   dataset_post_field='post',
                                   dataset_tag_field='theme_name',
                                   max_words=reliefweb_config.MAX_WORDS,
                                   # 20000, number of words to take from each post to tokenize),
                                   batch_size=reliefweb_config.BATCH_SIZE,
                                   epochs=reliefweb_config.EPOCHS,
                                   train_percentage=reliefweb_config.TRAINING_PERCENTAGE,  # 0.9
                                   skip_normalizing=reliefweb_config.FAST_TESTING
                                   )
    # RWModel['language'] = model_language
    RWModel['theme'] = model_theme

    import gc
    gc.collect()


@app.route("/")
# Instructions ENDPOINT
@cross_origin()
def main():
    return "Please, use the /tag endpoint with the param url to tag a url or pdf. Example: http://IP:PORT/tag?url=URL_WITH_HTTP"


@app.route("/tag")
# sample http://localhost:5000/tag?url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
@cross_origin()
def RWtag():
    import gc

    gc.collect()
    sample = request.args.get('url')
    # if (RWModel.get('language', '') == '') or (RWModel.get('theme', '') == ''):
    if RWModel.get('theme', '') == '':
        init()
    json_data = reliefweb_predict.url_to_tagged_json(model=RWModel, url=sample, threshold=reliefweb_config.THRESHOLD,
                                                     diff_terms=reliefweb_config.DIFF_TERMS_THRESHOLD)
    print("\nDone prediction for: " + sample)
    return json_data


@app.route("/html")
@cross_origin()
def htmlpage():
    return app.send_static_file('rwtag.html')


@app.route("/test")
def test():
    return "TEST ENDPOINT"


if __name__ == '__main__':
    import gc

    gc.collect()

    # get public IP -- if needed
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    publicIP = s.getsockname()[0]
    s.close()

    # app.run(debug=reliefweb_config.DEBUG, host=publicIP, port=reliefweb_config.PORT)  # use_reloader=False
    init()
    app.run(debug=reliefweb_config.DEBUG, host='0.0.0.0')  # use_reloader=False // This does not call to main
