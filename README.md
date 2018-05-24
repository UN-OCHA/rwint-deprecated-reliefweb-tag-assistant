# reliefweb-tag-assistant
ReliefWeb Tag Assistant - Tag urls using ReliefWeb tags

More information in [this presentation](https://docs.google.com/presentation/d/1p2t0mKYdAgVPQdC6cfNcnKIv_R3INUd0duuhaCGiq9A)

## Structure

- rwtag.html - *HTML page calling the main API endpoint and displaing the JSOn in a friendly way* 
- setup.py - *Python steup*
- reliefweb_tag/__init_.py - *Main program intializing the models and the API service*
- reliefweb_tag/relieweb_config.py - *Setting for the model and location of datasets*
- reliefweb_tag/reliefweb_ml_model.py - *Machine learning models for some of the fields/tags*
- reliefweb_tag/reliefweb_predict.py - *Main methods for tagging a url using different libraries*
- reliefweb_tag/reliefweb_tag_aux.py - *Additional functions for working with urls, files and strings*

## To install and execute in background
Requirements:
- python >3.5 - Available from the [Python Homepage](https://www.python.org/)
- Language model files: list of terms and dataset in a local route in csv format
- Themes model files: list of terms and dataset in a local route in csv format. 
 Recommended to use the report_theme_uneven_multiple-30k.csv dataset

*Sample datasets available for downloading in this [folder](https://drive.google.com/drive/folders/1Bo5B4DjtCH-tUOretNQmZvvX90bUsdwd?usp=sharing) and in the data directory*

```
# if you install from your home path, there is no need to change the config file
$ git clone https://github.com/reliefweb/reliefweb-tag-assistant/
$ gedit reliefweb-tag-assistant/reliefweb_config.py # configure the data main path and names of the datasets and vocabulary files.
$ cd reliefweb-tag-assistant
$ sudo apt-get install python3-setuptools
$ sudo python3 setup.py install
$ cd reliefweb_tag
$ python3 __init__.py &
```
