import reliefweb_config

def get_pdf_url (pdf_url):
    # libraries: textract (many bin formats / needs chardet 2.3.0 and there is a new version for other modules)
    # pyPDF2 , read page by page
    import requests
    import slate

    response = requests.get(pdf_url, stream=True)

    # with open(PATH_PDF_TMP_FILE, 'wb') as fd:
    fd = open(reliefweb_config.TMP_PDF_FILE, 'wb')
    fd.write(response.content)
    fd.close()

    # with open(PATH_PDF_TMP_FILE, "rb") as f:
    f =  open(reliefweb_config.TMP_PDF_FILE, "rb")
    doc = slate.PDF(f)
    f.close()
    return doc

def normalize (text):
    # normalizing the input -- it is supposed to remove stopwords (if not, nltk.corpus.stopwords.words()-- list of stopwords ) /
    # markup cleaning / new lines / punctuation and " (string.punctuation() ) / 's / to_lowercase / / names? / numbers - change to . or other char (#) / steaming (normalizing) - nltk.stem.porter.PorterStemmer()
    # remove if length <= 3 or >= 20 or contains http / roman numbers / bullet point (.) / text + number o al reves > text / ' / - /
    # for normalizing only necessary stop words (overrepresented), low caps, numbers by # / punctuation
    # REMOVE PREPOSITIONS N ALL LANGUAGES
    from cucco import Cucco
    import re

    cucco = Cucco()
    text = text.lower()
    text = cucco.normalize(text)

    text = re.sub('(\d+)%', '%' , text) # convert numbers percent to %
    text = re.sub('(\d+)', '#' , text) # convert numbers to #
    text = re.sub('(•|“|‘|”|’s|(.|)’)', "", text) # remove dot point for lists and “‘”
    # remove english possessive 'sop’s' and its
    # remove french l’ and d’ or ‘key
    #   Mr.Ging > mrging 'genderbasedviolence'  ascertain iraniansupported fuelefficient logisticsrelated
    # 19 471  no in 780 996 00 10pm a as 425 abovementioned avenirstrongp  genderrelated
    # in word_counts there are numbers and short words
    text = re.sub('#(?P<word>([a-zA-Z])+)', '\g<word>', text) # remove numbers before and after strings'
    text = re.sub('(?P<word>([a-zA-Z])+)#', '\g<word>', text) # remove numbers before and after strings'
    text = text.split()
    text = [w for w in text if ( len(w) > 2 and len(w) < 20 ) ] # remove short and very long words
    text = ' '.join(text)

    return text


def normalize2(text):
    # normalizing the input -- it is supposed to remove stopwords (if not, nltk.corpus.stopwords.words()-- list of stopwords ) /
    # markup cleaning / new lines / punctuation and " (string.punctuation() ) / 's / to_lowercase / / names? / numbers - change to . or other char (#) / steaming (normalizing) - nltk.stem.porter.PorterStemmer()
    # remove if length <= 3 or >= 20 or contains http / roman numbers / bullet point (.) / text + number o al reves > text / ' / - /
    # for normalizing only necessary stop words (overrepresented), low caps, numbers by # / punctuation
    # REMOVE PREPOSITIONS N ALL LANGUAGES
    from cucco import Cucco
    import re

    text = text.lower()

    cucco = Cucco(language=detect_language(text))
    normalizations = [
        'remove_stop_words',
        # 'remove_accent_marks', # french accents
        ('replace_hyphens', {'replacement': ' '}),
        ('replace_symbols', {'replacement': ' '}),
        ('replace_punctuation', {'replacement': ' '}),
        'remove_extra_white_spaces',
    ]
    text = cucco.normalize(text, normalizations)

    text = re.sub('(\d+)%', '%', text)  # convert numbers percent to %
    text = re.sub('(\d+)', '#', text)  # convert numbers to #
    # text = re.sub('#(?P<word>([a-zA-Z])+)', '\g<word>', text) # remove numbers before and after strings'
    # text = re.sub('(?P<word>([a-zA-Z])+)#', '\g<word>', text) # remove numbers before and after strings'
    text = text.split()
    # text = [w for w in text if ( len(w) > 2 and len(w) < 20 ) ] # remove short and very long words
    text = ' '.join(text)

    return text


def detect_language(text):
    from langdetect import detect
    return detect(text)
