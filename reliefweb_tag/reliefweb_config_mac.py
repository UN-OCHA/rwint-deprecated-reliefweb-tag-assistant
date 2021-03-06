# PORT for the API
PORT = 80

# Debug mode for the messages
DEBUG = True
FAST_TESTING = False  # If true, the input dataset is not processed nor normalized for quick testing
ALL_FIELDS = False  # If True, it will return all fields, if False, only the ones necessary for jobs and RW platform

# Name and route of the temporarary file to create for the PDFs
# TMP_PDF_FILE = "~/reliefweb-tag-assistant/temp/online_document.pdf"
TMP_PDF_FILE = "/Users/Miguel/reliefweb-tag-assistant/temp/online_document.pdf"  # config for Heroku

MODEL_PATH = "/Users/Miguel/reliefweb-tag-assistant/model/"
# Files with data (with read access)
# DATA_PATH = "~/reliefweb-tag-assistant/data/"
DATA_PATH = "/Users/Miguel/reliefweb-tag-assistant/data/"  #config for Heroku

MODEL_NAMES = ["job-type", "job-experience", "job-theme",
               "job-category"]  # array with all the models that we want to load  - ["theme", "job-type"]
CONFIG_ARRAY = ["max_words", "batch_size", "epochs", "train_percentage", "threshold", "diff_terms", "scope"]
MODEL_DEF = {}

'''
The vocabulary file is a text file with one row per possible value of the vocabulary. It has a header which should be
named the same as the vocabulary_name
The dataset should be a csv file with headers. The model will read the fields "post" and "value" which are mandatory.
'''

MODEL_DEF["theme"] = {"vocabulary": "rw-themes.csv",
                      "dataset": "report_theme_uneven_multiple-30k.csv",
                      "scope": "report",
                      # "dataset": "report_theme_en-1k.csv",
                      }
MODEL_DEF["language"] = {"vocabulary": "rw-languages.csv",
                         "dataset": "report_language-1k.csv"}
# To access a route : DATA_PATH + DATASETS["theme"]["vocabulary"]
MODEL_DEF["job-type"] = {"vocabulary": "rw-job-type.csv",
                         "dataset": "tmp-rw-job-type-2k-even.csv",
                         "scope": "job",
                         "batch_size": 128,  # 1024
                         # "dataset": "report_theme_en-1k.csv",
                         }
MODEL_DEF["job-experience"] = {"vocabulary": "rw-job-experience.csv",
                               "dataset": "tmp-rw-job-experience-4k-even.csv",
                               "scope": "job",
                               "batch_size": 256,  # 1024
                               # "dataset": "report_theme_en-1k.csv",
                               }
MODEL_DEF["job-theme"] = {"vocabulary": "rw-job-theme.csv",
                          "dataset": "tmp-rw-job-theme-10k-even.csv",
                          "scope": "job",
                          "batch_size": 512,  # 1024
                          # "dataset": "report_theme_en-1k.csv",
                          }
MODEL_DEF["job-category"] = {"vocabulary": "rw-job-category.csv",
                             "dataset": "tmp-rw-job-category-5k-even.csv",
                             "scope": "job",
                             "batch_size": 256,  # 1024
                             # "dataset": "report_theme_en-1k.csv",
                             }



# Neural network model parameters
MODEL_DEF["default"] = {
                      "max_words": 16384,  # 16384
                      "batch_size": 1024,  # 1024
                      "epochs": 4,
                      "train_percentage": 0.99,
                      "threshold": 0.1,
                      # When predicting, if a terms is lower than this percentage, it won't be returned
                      "diff_terms": 0.1,  # 0.01 # When predicting, if the difference with the previous
                      # predicted term is less than this, the term WILL appear
                      "scope":"all"
                      }