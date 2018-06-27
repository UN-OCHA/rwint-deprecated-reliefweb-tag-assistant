# Initializing the model
import reliefweb_config
import reliefweb_ml_model
import reliefweb_predict

# from https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py
# -*- coding: utf-8 -*-
"""
Downloads the necessary NLTK models and corpora required to support
all of newspaper's features. Modify for your own needs.
"""
import nltk

REQUIRED_CORPORA = [
    'brown',  # Required for FastNPExtractor
    'punkt',  # Required for WordTokenizer
    'maxent_treebank_pos_tagger',  # Required for NLTKTagger
    'movie_reviews',  # Required for NaiveBayesAnalyzer
    'wordnet',  # Required for lemmatization and Wordnet
    'stopwords'
]

for each in REQUIRED_CORPORA:
    print(('Downloading "{0}"'.format(each)))
    nltk.download(each)
print("Finished.")

print("> Initializing machine learning model")
RWModel = reliefweb_ml_model.ReliefwebModel()

print("> MAIN: Creating neural network for language")
# as the language model is accurate, we reduce its complexity for gaining on memory
RWModel.create_train_model(vocabulary_name='language',
                           vocabulary_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["language"][
                               "vocabulary"],
                           # path+"rw-languages.csv",
                           dataset_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["language"]["dataset"],
                           # path+'report_language-1k.csv',
                           term_field='language',
                           vocabulary_language='',  # values: English, Spanish, French, ''
                           dataset_post_field='post',
                           dataset_tag_field='language_name',
                           max_words=reliefweb_config.MAX_WORDS,
                           # 20000, number of words to take from each post to tokenize),
                           batch_size=reliefweb_config.BATCH_SIZE_LANG,
                           epochs=reliefweb_config.EPOCHS,
                           train_percentage=reliefweb_config.TRAINING_PERCENTAGE,  # 0.9
                           skip_normalizing=reliefweb_config.FAST_TESTING
                           )
print("> MAIN: Creating neural network for themes")
RWModel.create_train_model(vocabulary_name='theme',
                           vocabulary_language='',  # values: English, Spanish, French, ''
                           vocabulary_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["theme"][
                               "vocabulary"],  # path+"rw-themes.csv",
                           dataset_file=reliefweb_config.DATA_PATH + reliefweb_config.DATASETS["theme"]["dataset"],
                           # path+'report_theme_uneven_multiple-30k.csv',#'report_theme_en-1k.csv',#
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

import gc
gc.collect()

# Creating the API endpoints
from flask import Flask, request

app = Flask(__name__)
app.debug = False
app.threaded = False
# two parameters to avoid "Tesorflow ... is not an element of this graph"

# get public IP
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
publicIP = s.getsockname()[0]
s.close()


@app.route("/")
# Instructions ENDPOINT
def main():
    return "Please, use the /tag endpoint with the param url to tag a url or pdf. Example: http://IP:PORT/tag?url=URL_WITH_HTTP"


@app.route("/tag")
# sample http://localhost:5000/tag?url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
def RWtag():
    import gc

    gc.collect()
    sample = request.args.get('url')
    json_data = reliefweb_predict.url_to_tagged_json(model=RWModel, url=sample, threshold=reliefweb_config.THRESHOLD,
                                                     diff_terms=reliefweb_config.DIFF_TERMS_THRESHOLD)
    return json_data


if __name__ == '__main__':
    import gc

    gc.collect()
    app.run(debug=reliefweb_config.DEBUG, host=publicIP, port=reliefweb_config.PORT)  # use_reloader=False

if __name__ == '/tag':
    import gc

    gc.collect()
    app.run(debug=reliefweb_config.DEBUG, host=publicIP, port=reliefweb_config.PORT)  # use_reloader=False
