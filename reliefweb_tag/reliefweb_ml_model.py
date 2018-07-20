# TODO : Normalization improvements
# TODO : Auto-training from API
"""
This module creates a model based on a training dataset (input) and a given vocabulary (output)

DATASET FORMAT: title + body, prim_country, country, source, theme, format, OCHA product, disaster, disaster type,
vulnerable group, language
 vocabulary terms with id
 To create the queries: https://stackoverflow.com/questions/12113699/get-top-n-records-for-each-group-of-grouped-results

Specs of the entry
- The dataset is a csv field
- The dataset should be balanced (same number of entries per "tag")
-  There should be a header with no spaces, the name of the entry and the output fields should be specified when calling
this module: dataset_post_field and dataset_tag_field

Based on https://cloud.google.com/
blog/big-data/2017/10/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts

REQUIREMENTS:
- pip install https://github.com/timClicks/slate/archive/master.zip #slate
"""

from __future__ import division
from __future__ import print_function

# Debugging options
import logging
import os.path
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import utils
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder

from reliefweb_tag import reliefweb_config
from reliefweb_tag import reliefweb_tag_aux

if reliefweb_config.DEBUG:
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def read_data(dataset_file):
    # read the dataset
    # return data object with the input

    logging.debug('START: Reading the input / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    data = pd.read_csv(dataset_file)
    data.head()

    logging.debug('END: Reading the input / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return data


class ReliefwebModel:
    # ReliefwebModel is an array of models and more attributes on its root.

    def __init__(self):

        self.model = {}
        self.text_labels = {}
        pd.options.mode.chained_assignment = None  # don't display warnings from pandas
        return

    # main method for creating the whole model:
    # creates the vocabulary, normalize the training set, create the dataset for training,
    # trains and sets the new final model on the models dictionary
    def create_train_one_tag_model(self,
                           vocabulary_name,
                           vocabulary_file,
                           dataset_file,
                           model_path,
                           term_field='value',
                           vocabulary_language='English',  # values: English, Spanish, French
                           dataset_post_field='post',
                           dataset_tag_field='value',
                           max_words=16384,  # 20000, number of words to take from each post to tokenize),
                           batch_size=128,
                           epochs=5,
                           train_percentage=0.9,
                           skip_normalizing=False
                           ):
        """
        read_vocabulary(vocabulary_name)
        normalize_input()
          read_data()
          normalize
        prepare_dataset() -> data
           process_input()
        create_model (vocabulary_name)-> model
        """

        # if model file exists -> load model
        # if not, create model and save

        print("Looking for file  " + model_path + "model_" + vocabulary_name + "_*.*")

        import pickle # to load and save the tokenizer

        model = Sequential()

        if os.path.isfile(model_path + "model_" + vocabulary_name + "_model.json") and \
           os.path.isfile(model_path + "model_" + vocabulary_name + "_weights.h5") and \
           os.path.isfile(model_path + "model_" + vocabulary_name + "_tokenizer.pickle"):
            # read the model if it exists

            tic = time.time()

            self.read_vocabulary(vocabulary_file, term_field)

            toc = time.time()
            print ("(t) read_vocabulary : %d " % (toc-tic))
            tic = time.time()

            data = self.normalize_input(dataset_file,
                                        dataset_post_field,
                                        dataset_tag_field,
                                        skip_normalizing=True)
            toc = time.time()
            print ("(t) normalize_input : %d " % (toc-tic))
            tic = time.time()

            self.prepare_dataset(data, vocabulary_name, max_words,
                                 train_percentage)  # 20000, number of words to take from each post to tokenize)

            toc = time.time()
            print ("(t) prepare_dataset : %d " % (toc-tic))
            tic = time.time()

            # load json and create model
            json_file = open(model_path + "model_" + vocabulary_name + "_model.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(model_path + "model_" + vocabulary_name + "_weights.h5")
            # model = load_model(model_path + "model_" + vocabulary_name + ".model")

            # loading
            tokenizer_file = open(model_path +'model_' + vocabulary_name + '_tokenizer.pickle', 'rb')
            self.tokenize = pickle.load(tokenizer_file)

            print("Loaded model " + vocabulary_name + " from disk")

        else:

            self.read_vocabulary(vocabulary_file, term_field)
            data = self.normalize_input(dataset_file,
                                        dataset_post_field,
                                        dataset_tag_field,
                                        skip_normalizing)
            self.prepare_dataset(data, vocabulary_name, max_words,
                                 train_percentage)  # 20000, number of words to take from each post to tokenize)
            model = self.create_one_tag_model(batch_size, epochs)
            # save model

            # model.save(model_path + "model_" + vocabulary_name + ".model")
            # serialize weights to HF5
            model.save_weights(model_path + "model_" + vocabulary_name + "_weights.h5")
            # serialize model to JSON
            model_json = model.to_json()
            json_file = open(model_path + "model_" + vocabulary_name + "_model.json", "w+")
            json_file.write(model_json)
            json_file.close()
            # save tokenizer
            import pickle
            # saving tokenizer (vector of words to map the input text)
            tokenizer_file = open(model_path + 'model_' + vocabulary_name + '_tokenizer.pickle', 'w+b')
            pickle.dump(self.tokenize, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

            print("Saved model " + vocabulary_name + " to disk")

        # this is key : save the graph after loading the model
        self.graph = tf.get_default_graph()
        self.model = model

    # pure machine learning model: builds the model, trains it and validates it
    def create_one_tag_model(self,
                     batch_size=128,
                     epochs=5,
                     ):
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

        model = self.build_one_tag_model()
        model = self.train_one_tag_model(model)
        model = self.validate_model(model)

        single_post_serie = pd.Series(["First prediction to initialize the model for multi-threading"])
        single_test = self.tokenize.texts_to_matrix(single_post_serie)
        model.predict(single_test)

        single_post_serie = pd.Series(["First prediction to initialize the model"])
        single_test = self.tokenize.texts_to_matrix(single_post_serie)
        model.predict(single_test)

        return model

    def read_vocabulary(self, vocabulary_file, term_field):
        # read the dataset
        # return data object with the input

        logging.debug('START: Reading the vocabulary file / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        data = pd.read_csv(vocabulary_file)
        data.head()

        self.vocabulary_terms = data[term_field][:len(data)]
        self.num_classes = len(data)

        return

    def normalize_input(self,
                        dataset_file,
                        dataset_post_field,
                        dataset_tag_field,
                        skip_normalizing=False):

        self.dataset_post_field = dataset_post_field
        self.dataset_tag_field = dataset_tag_field

        data = read_data(dataset_file)

        logging.info(
            'START: normalize_input - %d entries / ' % len(data) + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        tic_input = time.time()

        tic = time.time()

        if not skip_normalizing:
            for i in range(1, len(data)):
                data[dataset_post_field][i] = reliefweb_tag_aux.normalize_global(data[dataset_post_field][i])
                data[dataset_tag_field][i] = data[dataset_tag_field][i].strip(' \t\n\r')
                # TODO: Tokenizer later theoretically does this
                if i % 1000 == 0:
                    # Displaying time left
                    toc = time.time()
                    logging.debug("normalize_input - %d entries in %d seconds / Left estimation: < %d minutes" % (
                        i, (toc - tic), ((toc - tic) * (len(data) - i)) / (i * 60) + 1))

        toc_input = time.time()
        logging.info("END: normalize_input / %d sec Elapsed" % (toc_input - tic_input))

        return data

    def prepare_dataset(self, data, vocabulary_name, max_words=16384, train_percentage=0.9):

        logging.info('START: prepare_dataset / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_size = int(len(data) * train_percentage)

        # creating the training and testing sets
        train_posts = data[self.dataset_post_field][:train_size]
        train_tags = data[self.dataset_tag_field][:train_size]

        test_posts = data[self.dataset_post_field][train_size:]
        test_tags = data[self.dataset_tag_field][train_size:]

        self.MAX_WORDS = max_words

        self.tokenize = text.Tokenizer(num_words=self.MAX_WORDS, char_level=False, lower=True)
        logging.info("prepare_dataset - END: input tokenizer")

        self.tokenize.fit_on_texts(train_posts)  # only fit on train / build the index of words -- top MAX_WORDS
        self.x_train = self.tokenize.texts_to_matrix(train_posts) # create a matrix with one item per document
        self.x_test = self.tokenize.texts_to_matrix(test_posts)
        # With  mode="count" the results are not better
        logging.info("prepare_dataset - END: tokenize.fit_on_texts & tokenize.texts_to_matrix")

        # Use sklearn utility to convert label strings to numbered index
        encoder = LabelEncoder() # Encode labels with value between 0 and n_classes-1.
        encoder.fit(self.vocabulary_terms)
        self.y_train = encoder.transform(train_tags)
        self.y_test = encoder.transform(test_tags)
        self.text_labels = encoder.classes_
        logging.info("prepare_dataset - END: encoder transform / " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Converts the labels to a one-hot representation
        self.num_classes = np.max(self.y_train) + 1
        self.y_train = utils.to_categorical(self.y_train, self.num_classes) # class vector (integers) to binary class matrix
        self.y_test = utils.to_categorical(self.y_test, self.num_classes)
        logging.info(
            "prepare_dataset - END: to_categorical conversion / " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        logging.info('  x_train shape: ' + str(self.x_train.shape))
        logging.info('  x_test shape: ' + str(self.x_test.shape))
        logging.info('  y_train shape: ' + str(self.y_train.shape))
        logging.info('  y_test shape: ' + str(self.y_test.shape))
        logging.info('END: prepare_dataset / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        return

    def build_one_tag_model(self):

        logging.debug('START: build_model / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # BUILD THE MODEL
        model = Sequential()
        model.add(Dense(512, input_shape=(self.MAX_WORDS,)))  # hidden layers neurons: 512
        model.add(Activation('relu'))
        model.add(Dropout(0.2))  # Default was to 0.5 ** 0.2 seems better results
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logging.debug('END: build_model / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return model

    def train_one_tag_model(self, model):
        logging.debug('START: train_model / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS,
                            verbose=1,
                            validation_split=0.1)

        logging.debug('END: train_model / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return model

    def validate_model(self, model):
        logging.debug('START: validate_model / ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        score = model.evaluate(self.x_test, self.y_test,
                               batch_size=self.BATCH_SIZE, verbose=1)
        logging.info('Test score: ' + str(score[0]))
        logging.info('Test accuracy: ' + str(score[1]))

        logging.debug("END: validate_model / " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return model

    def predict_nonlanguage_text(self,
                                 sample,
                                 vocabulary_name,
                                 threshold=0.5,
                                 diff_terms=0.05):
        graph = self.graph
        with graph.as_default():

            result = []

            if self.model is None:
                logging.ERROR("ERROR: The unique model for vocabulary '%s' has not been defined yet" % vocabulary_name)
                return result
            else:
                result = self.predict_one_tag_value(self.model, vocabulary_name, sample, threshold,
                                            diff_terms)
            return result

    def predict_one_tag_value(self, model, vocabulary_name, sample,
                      threshold=0.5,
                      diff_terms=0.05):
        """
        Prediction function

        :param model:
        :param vocabulary_name:
        :param sample:
        :param threshold:
        :param diff_terms:
        :return: array with all the tags and confidence until confidence is less than threshold. If there is no value
        with that condition, it returns the maximum value.
        if the next values have a different less than diff_terms, it also returns those values
        """

        graph = self.graph
        with graph.as_default():

            sample = reliefweb_tag_aux.normalize_global(sample)
            single_post_serie = pd.Series([sample])
            single_test = self.tokenize.texts_to_matrix(single_post_serie)
            prediction = model.predict(single_test)
            prev_predicted_confidence = 1
            predicted_label = ""
            predicted_confidence = 1
            result = {}

            while ((predicted_confidence > threshold) or
                   (((prev_predicted_confidence - predicted_confidence) / predicted_confidence) < diff_terms)):
                if len(result) > 0:
                    prev_predicted_confidence = float(result[predicted_label])
                predicted_confidence = prediction[0, np.argmax(prediction)]
                predicted_label = self.text_labels[np.argmax(prediction)]
                result[predicted_label] = str(predicted_confidence)
                prediction[0, np.argmax(prediction)] = 0

            # order the result by %

            from operator import itemgetter
            result = sorted(result.items(), key=itemgetter(1), reverse=True)

            return result

    def predict_language(self, sample):

        graph = self.graph
        with graph.as_default():
            model = self.model

            self.tokenize = text.Tokenizer(num_words=self.MAX_WORDS, char_level=False, lower=True)

            sample = reliefweb_tag_aux.normalize_global(sample)
            single_post_serie = pd.Series([sample])
            self.tokenize.fit_on_texts(single_post_serie)  # only fit on train
            single_test = self.tokenize.texts_to_matrix(single_post_serie)
            prediction = model.predict(single_test)

            predicted_confidence = prediction[0, np.argmax(prediction)]
            predicted_label = self.text_labels[np.argmax(prediction)]

            # print ("DEBUG -- language: " +sample + " / " + str(single_test) + "/"+ predicted_label + " / "
            # + str(predicted_confidence) + " /  " + str(self.text_labels['language']) + " / " + str(prediction))

            return {predicted_label: str(predicted_confidence)}

    # def predict_multilingual_text(self,
    #                               sample,
    #                               vocabulary_name,
    #                               threshold=0.5,
    #                               diff_terms=0.05):
    #     graph = self.graph
    #     with graph.as_default():
    #
    #         result = []
    #         if self.model is None:
    #             logging.ERROR("ERROR: The language model has not been defined yet")
    #             return result
    #         else:
    #             language = self.predict_language(sample)[0][0]
    #             # TODO: If there is no value this return an error and not "None"
    #             if self.model is None:
    #                 logging.ERROR("ERROR: The model for vocabulary '%s' and language '%s' has not been defined yet" %
    #                     ( vocabulary_name, language))
    #                 return result
    #             else:
    #                 result = self.predict_value(self.model, vocabulary_name, sample, threshold,
    #                                             diff_terms)
    #         return result

