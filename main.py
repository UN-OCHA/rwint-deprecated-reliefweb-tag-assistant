# Initializing the model
import socket

from flask import Flask, request
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

    print("> MAIN: Creating neural network for themes")

    if models.get('theme', '') == '':
        model_theme = reliefweb_ml_model.ReliefwebModel("theme")
        models['theme'] = model_theme


# Creating the API endpoints
@app.route("/")
# Instructions ENDPOINT
@cross_origin()
def main():
    return "Please, use the /tag endpoint with the param url to tag a url or pdf. Example: http://IP:PORT/tag?url=URL_WITH_HTTP"


@app.route("/tag_url")
# sample http://localhost:5000/tag_url?url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
@cross_origin()
def reliefweb_tag_url():
    import gc

    gc.collect()
    url = request.args.get('url')
    # if (RWModel.get('language', '') == '') or (RWModel.get('theme', '') == ''):
    init_models()
    json_data = reliefweb_predict.predict(models=models, _input=url)
    print("\nDone prediction for: " + url)
    return json_data


@app.route("/tag_text")
# sample http://localhost:5000/tag_text?text=Blablalblbalblalbalñldfjk
@cross_origin()
def reliefweb_tag_text():
    import gc

    gc.collect()
    text = request.args.get('text')
    init_models()
    json_data = reliefweb_predict.predict(models=models, _input=text)
    print("\nDone prediction for: " + str(text)[:20] + "...")
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
    init_models()
    app.run(debug=reliefweb_config.DEBUG, host='0.0.0.0')  # use_reloader=False // This does not call to main
