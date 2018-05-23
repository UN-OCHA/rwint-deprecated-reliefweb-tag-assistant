# PORT for the API
PORT = 5463

# Debug mode for the messages
DEBUG = False

# Name and route of the temporarary file to create for the PDFs
# TMP_PDF_FILE = '/home/deeplearning/Desktop/reliefweb deep learning/online_document.pdf'
TMP_PDF_FILE = "C:\DeepLearning\final_data"
# PATH_PDF_TMP_FILE

# Files with data (with read access)
# DATA_PATH = "/home/deeplearning/Desktop/reliefweb deep learning/"
DATA_PATH = "C:\DeepLearning\final_data"
DATASETS = {}
DATASETS["theme"] =  {"vocabulary":"rw-themes.csv",
                        # "dataset":"report_theme_uneven_multiple-30k.csv" } ,
                      "dataset":"report_theme_en-1k.csv" }
DATASETS["language"] =  {"vocabulary":"rw-languages.csv",
                         "dataset":"report_language-1k.csv"}
# To access a route : DATA_PATH + DATASETS["theme"]["vocabulary"]

# Neural netowrk model parameters
MAX_WORDS = 16384 # 16384
BATCH_SIZE_LANG = 128
BATCH_SIZE = 1024 # 1024
EPOCHS = 4
TRAINING_PERCENTAGE = 0.99
THRESHOLD = 0.1
# When predicting, if a terms is lower than this percentage, it won't be returned
DIFF_TERMS_THRESHOLD =  0.1 # 0.01
# When predicting, if the difference with the previous predicted term is less than this, the term WILL appear
