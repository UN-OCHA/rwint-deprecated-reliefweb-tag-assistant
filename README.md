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
Requirement: > python 3.5 - Available from the [Python Homepage](https://www.python.org/)

```
$ git clone https://github.com/reliefweb/reliefweb-tag-assistant/
$ cd reliefweb-tag-assistant
$ sudo python3.5 setup.py install
$ python3.5 __init.py__ &
```
