# reliefweb-tag-assistant
ReliefWeb Tag Assistant - Tag urls using ReliefWeb tags

More information in [this presentation](https://docs.google.com/presentation/d/1p2t0mKYdAgVPQdC6cfNcnKIv_R3INUd0duuhaCGiq9A)

## Structure

- rwtag.html - *HTML page calling the main API endpoint and displaing the JSOn in a friendly way* 
- setup.py - *Python setup for setuptools*
- reliefweb_tag/__init__.py - *Main program intializing the models and the API service*
- reliefweb_tag/relieweb_config.py - *Setting for the model and location of datasets*
- reliefweb_tag/reliefweb_ml_model.py - *Machine learning models for some of the fields/tags*
- reliefweb_tag/reliefweb_predict.py - *Main methods for tagging a url using different libraries*
- reliefweb_tag/reliefweb_tag_aux.py - *Additional functions for working with urls, files and strings*

## Requirements

- python >3.5 - Available from the [Python Homepage](https://www.python.org/)
- Language model files: list of terms and dataset in a local route in csv format
- Themes model files: list of terms and dataset in a local route in csv format. *Recommended to use the report_theme_uneven_multiple-30k.csv dataset*

*Sample datasets available for downloading in this [folder](https://drive.google.com/drive/folders/1Bo5B4DjtCH-tUOretNQmZvvX90bUsdwd?usp=sharing) and in the data directory*

## Installing and executing in Ubuntu

- Create and run virtual environment

```
$ virtualenv testenv --python=/usr/bin/python3.6 
$ source testenv/bin/activate
```

- Modules required

 setuptools

```
$ sudo apt-get install python3-setuptools
```

 pip for python 3 and newspaper3k. Manual installation due to this [issue](https://github.com/codelucas/newspaper/issues/350)
 > Already in requirements.txt as an external resource

```
$ sudo apt-get install python3-pip
$ pip install -r requirements.txt

```

- Main reliefweb-tag 

```
# if you install from your home path, there is no need to change the config file
$ git clone https://github.com/reliefweb/reliefweb-tag-assistant/
$ gedit reliefweb-tag-assistant/reliefweb_config.py # configure the data main path and names of the datasets and vocabulary files.
$ cd reliefweb-tag-assistant
$ sudo python3 setup.py install
$ python3 main.py &
```

Once the backend is running you can use the following endpoints:

- http://localhost:5000/tag_url?scope=report&url=https://stackoverflow.com/questions/24892035/python-flask-how-to-get-parameters-from-a-url
- http://localhost:5000/tag_text?scope=job&text=Blablalblbalblalbal%C3%B1ldfjk

(Updated information and last names for the endpoints in the ```main.py``` file)

Also, as a frontend, there are 2 html webpages that can be loaded in any browser that call those endpoints automatically and display the results in a more friendly way. Those 2 html files have a call to the endpoint that should be modified to reflect your current configuration.

