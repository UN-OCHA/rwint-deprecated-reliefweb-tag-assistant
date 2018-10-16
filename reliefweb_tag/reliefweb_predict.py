def process_url_input(_input):
    """
    Gets the url and sets the metadata returning a dict including basic metatags, geolocation and language
    :param _input: url of a page or a pdf
    :return: dictionary with the basic tagging
    """

    from reliefweb_tag import reliefweb_tag_aux

    try:
        if not ((_input.lower()[:7] == 'http://') or (_input.lower()[:8] == 'https://')):
            # if it is not a URL
            return {"error": "The input should be an url", "full_text": ""}

        sample_dict = tag_metadata(_input)
        if reliefweb_tag_aux.reliefweb_config.ALL_FIELDS:
            tag_language_langdetect(sample_dict)
        tag_geolocation(sample_dict)

    except Exception as e:
        return {"error": str(e), "full_text": ""}

    return sample_dict


def process_text_input(_input):
    """
    Gets the text and sets the metadata returning a dict including basic metatags, geolocation and language
    :param _input: plan text to predict
    return: dictionary with the basic tagging
    """

    try:
        sample_dict = tag_metadata_text(_input)
        tag_language_langdetect(sample_dict)
        tag_geolocation(sample_dict)

    except Exception as e:
        return {"error": str(e), "full_text": ""}

    return sample_dict


def predict(_models, _sample_dict, _scope):
    """
    Main method to tag a URL or text, return a JSON with all the fields populated and predicted
    :param _models: array of machine learning models
    :param _sample_dict: dictionary to fill in with tags with 'full_text'
    :param _scope: scope to match for each ML model to tag on that vocabulary
    :return:
    """

    try:
        for each in _models.keys():
            if (_models[each].config['scope'] == _scope) or (_models[each].config['scope'] == "all"):
                tag_machine_learning_model(_models[each], _sample_dict)

    except Exception as e:
        _sample_dict = {"error": str(e), "full_text": ""}

    return _sample_dict


def tag_metadata(_input):
    """
    Gets all the tags from the newspaper library
    :param _input: can be a url or a text string
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux

    from newspaper import Article, Config

    configuration = Config()
    configuration.request_timeout = 15  # default = 7
    configuration.keep_article_html = True

    isurl = (_input.lower()[:7] == 'http://') or (_input.lower()[:8] == 'https://')

    if isurl:
        article = Article(_input, config=configuration)
    else:
        article = Article("")

    # if URL IS PDF or any binary then
    if _input.lower()[-4:] in [".pdf"]:
        try:
            pdf = reliefweb_tag_aux.get_pdf_url(input)
        except Exception as e:
            raise Exception(e)
        pdf_text = ' '.join(pdf)
        article.set_text(pdf_text)
        article.set_article_html(pdf_text)
        article.set_html(pdf_text)
        article.title = pdf.metadata[0].get('Title',
                                            '')  # set title fills the field with Configuration when title empty
        article.set_authors([pdf.metadata[0].get('Author', '')])
        article.publish_date = pdf.metadata[0].get('CreationDate', '')
        article.html

    if not isurl:
        article.download(input)
        article.text = input  # 0.2.2 - article.set_text(input)
        article.set_article_html(input)
        article.set_html(input)
        article.title = ""  # 0.2.2 - article.set_title(input)
        article.set_html(input)
    else:
        # if it is not pdf
        article.download()
        article.html
        article.parse()

    data = {'url': _input,
            'title': article.title,
            'text': article.text,
            'article_html': article.article_html}

    if article.article_html == '':
        article_html = str(article.html)  # takes all the html of the page
    else:
        article_html = article.article_html
    import html2text  # Other libraries are tomd and pandoc
    data['body_markdown'] = html2text.html2text(str(article_html))
    data['full_text'] = "{0} {1}".format(article.title, str(article.text))  # necessary for predictions

    if reliefweb_tag_aux.reliefweb_config.ALL_FIELDS:
        try:
            article.nlp()

            data['publish_date'] = str(article.publish_date)
            data['meta_lang'] = article.meta_lang
            data['meta_keywords'] = article.meta_keywords,
            data['topics'] = article.meta_data.get('TOPICS', '')
            data['language'] = article.meta_data.get('LANGUAGE', '')
            data['publication_type'] = article.meta_data.get('PUBLICATION_TYPE', '')
            data['authors'] = article.authors
            data['tags'] = list(article.tags)
            data['keywords'] = article.keywords
            data['summary'] = article.summary
            data['top_image'] = article.top_image
            data['article_html'] = article_html
        except Exception as e:
            print("There was an error processing the text, some of the metadata will not be available")
            raise e

    return data

def tag_metadata_text(_text):
    """
    Gets all the tags from the newspaper library
    This method provides less meta tags than the url method. Some tags as Author or pubData are not applicable.
    :param _text: text not starting as URL
    :return:
    """
    import html2text

    data = {'text': _text,
            'full_text': _text,
            'article_html': _text,
            'title': ""
            }
    data['body_markdown'] = html2text.html2text(str(data['article_html']))

    return data


def tag_machine_learning_model(_model, _dict_in):
    """
    Creates the 'theme' value on the dictionary based on the theme neural model
    :param _model:
    :param _dict_in:
    :return:
    """

    text = _dict_in['full_text']
    predicted_value = _model.predict_nonlanguage_text(sample=text)
    _dict_in[_model.vocabulary_name] = predicted_value
    return _dict_in


def tag_language_langdetect(_dict_in):
    """
    Creates the 'langdetect_language' value on the dictionary based on the theme neural model
    :param _dict_in:
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux
    _dict_in['langdetect_language'] = reliefweb_tag_aux.detect_language(_dict_in['full_text'])
    return _dict_in


def tag_geolocation(_dict_in):
    """
    Creates the 'countries', 'primary_country', 'countries_iso2', 'cities', 'nationalities' value on the dictionary
    using the GeoText module

    TODO: Geotagging with coordinates
    from geopy.geocoders import Nominatim
    geolocator = Nominatim()
    location = geolocator.geocode("175 5th Avenue NYC")
    print(location.address, location.latitude, location.longitude))

    :param _dict_in:
    :return:
    """

    from geotext import GeoText
    import pycountry
    from collections import Counter

    places = GeoText(_dict_in['full_text'])

    countries_frequency = Counter(places.country_mentions)
    cities_frequency = Counter(places.cities)  # It doesn't report the frequency of the cities
    # TODO: To use https://github.com/ushahidi/geograpy for better Geo Performance

    countries_iso2 = list(countries_frequency)  # list of unique elements
    cities = list(cities_frequency)

    _dict_in['cities'] = cities
    _dict_in['nationalities'] = places.nationalities
    _dict_in['countries_iso2'] = countries_iso2

    # primary city
    if len(cities_frequency) > 0:
        primary_city = cities_frequency.most_common(1)[0][0]
        _dict_in['primary_city'] = primary_city
    else:
        _dict_in['primary_city'] = ""

    # country and primary country
    i = 0
    _dict_in['countries'] = []
    while i < len(countries_iso2):
        country_iso2 = countries_iso2[i]
        i = i + 1
        if country_iso2 == 'UK':
            country_iso2 = 'GB'
        try:  # In case that the FIPS code is not recognized by pycountry
            country = pycountry.countries.get(alpha_2=country_iso2)
            _dict_in['countries'].append((country.name, country.alpha_3, country_iso2))
            if i == 0:  # primary country
                _dict_in['primary_country'] = [country.name, country.alpha_3]
        except Exception as e:
            _dict_in['countries'] = []
            _dict_in['error'] = 'A country identified ' + country_iso2 + ' has a FIPS code not matching any ISO2'
