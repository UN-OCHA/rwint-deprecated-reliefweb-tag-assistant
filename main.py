# Initializing the model
import socket

from flask import Flask, request
from flask import make_response
from flask_cors import CORS, cross_origin

from reliefweb_tag import reliefweb_ml_model, reliefweb_predict, reliefweb_config

global app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-type'
# Content-type: application/json
app.debug = False
app.threaded = False

global models
models = {}

print("** In the main flow **")


def init_models():
    print("Initializing the ReliefWeb Tag Assistant: auto-tag urls using RW Tags and Machine Learning")

    print("> Initializing machine learning model")

    for each in reliefweb_config.MODEL_NAMES:
        # TODO: What does this language collector means? Can we remove it?
        if models.get(each, '') == '':
            print("> MAIN: Creating neural network for " + each)
            model = reliefweb_ml_model.ReliefwebModel(each)
            models[each] = model


# Creating the API endpoints
@app.route("/")
# Instructions ENDPOINT
@cross_origin()
def main():
    return "Please, use the /tag endpoint with the param url to tag a url or pdf. Example: http://IP:PORT/tag?url=URL_WITH_HTTP"


@app.route("/tag_url")
# sample http://localhost:5000/tag_url?scope=report&url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
@cross_origin()
def reliefweb_tag_url():

    import gc
    import json

    gc.collect()
    url = request.args.get('url')
    scope = request.args.get('scope')
    # if (RWModel.get('language', '') == '') or (RWModel.get('theme', '') == ''):
    sample_dict = reliefweb_predict.process_url_input(url)
    init_models()
    if scope in ["report", "job"]:
        sample_dict = reliefweb_predict.predict(_models=models, _sample_dict=sample_dict, _scope=scope)
    else:
        sample_dict = {"error": "scope parameter should be job or report", "full_text": ""}
    print("\nDone prediction for: " + url)

    response = make_response(json.dumps(sample_dict, indent=4))
    response.headers['content-type'] = 'application/json'
    return response


@app.route("/tag_text", methods=['POST', 'GET'])
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

@app.route("/html")
@cross_origin()
def htmlpage():
    return app.send_static_file('rwtag.html')


@app.route("/test")
def test():
    return "TEST ENDPOINT"


if __name__ == '__main__':
    # get public IP -- if needed
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    publicIP = s.getsockname()[0]
    s.close()

    # app.run(debug=reliefweb_config.DEBUG, host=publicIP, port=reliefweb_config.PORT)  # use_reloader=False
    init_models()
    app.run(debug=reliefweb_config.DEBUG, host='0.0.0.0')  # use_reloader=False // This does not call to main
