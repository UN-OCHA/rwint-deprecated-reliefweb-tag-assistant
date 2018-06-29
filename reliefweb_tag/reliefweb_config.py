# PORT for the API
PORT = 80

# Debug mode for the messages
DEBUG = True
FAST_TESTING = False  # If true, the input dataset is not processed nor normalized for quick testing

# Name and route of the temporarary file to create for the PDFs
# TMP_PDF_FILE = "~/reliefweb-tag-assistant/temp/online_document.pdf"
TMP_PDF_FILE = "~/temp/online_document.pdf"  # config for Heroku


# Files with data (with read access)
# DATA_PATH = "~/reliefweb-tag-assistant/data/"
DATA_PATH = "~/data/"  #config for Heroku
DATASETS = {}
DATASETS["theme"] = {"vocabulary": "rw-themes.csv",
                     #"dataset": "report_theme_uneven_multiple-30k.csv"}
                     "dataset": "report_theme_en-1k.csv"}
DATASETS["language"] = {"vocabulary": "rw-languages.csv",
                        "dataset": "report_language-1k.csv"}
# To access a route : DATA_PATH + DATASETS["theme"]["vocabulary"]

# Neural netowrk model parameters
MAX_WORDS = 16384  # 16384
BATCH_SIZE_LANG = 128
BATCH_SIZE = 1024  # 1024
EPOCHS = 4
TRAINING_PERCENTAGE = 0.99
THRESHOLD = 0.1
# When predicting, if a terms is lower than this percentage, it won't be returned
DIFF_TERMS_THRESHOLD = 0.1  # 0.01
# When predicting, if the difference with the previous predicted term is less than this, the term WILL appear
