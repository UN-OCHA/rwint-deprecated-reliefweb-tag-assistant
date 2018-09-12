def process_url_input(_input):
    """
    Gets the url and sets the metadata returning a dict including basic metatags, geolocation and language
    :param _input: url of a page or a pdf
    :return: dictionary with the basic tagging
    """
    try:
        if not ((_input.lower()[:7] == 'http://') or (_input.lower()[:8] == 'https://')):
            # if it is not a URL
            return {"error": "The input should be an url", "full_text": ""}

        # if URL IS PDF or any binary then
        if _input.lower()[-4:] in [".pdf"]:
            sample_dict = tag_metadata_url_pdf(_input)
        else:
            sample_dict = tag_metadata(_input)
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

    try:
        article.parse()
        article.nlp()
    except Exception as e:
        print("There was an error processing the text, some of the metadata will not be available")

    data = {'url': _input,
            'publish_date': str(article.publish_date),
            'meta_lang': article.meta_lang,
            'meta_keywords': article.meta_keywords,
            'topics': article.meta_data.get('TOPICS', ''),
            'language': article.meta_data.get('LANGUAGE', ''),
            'publication_type': article.meta_data.get('PUBLICATION_TYPE', ''),
            'text': article.text,
            'full_text': article.title + " " + str(article.text),
            'article_html': article.article_html,
            'authors': article.authors,
            'title': article.title,
            'tags': list(article.tags),
            'keywords': article.keywords,
            'summary': article.summary,
            'top_image': article.top_image}

    if article.article_html == '':
        data['article_html'] = article.html  # takes all the html of the page

    import html2text  # Other libraries are tomd and pandoc
    data['body_markdown'] = html2text.html2text(str(data['article_html']))
    data['article_html'] = 'Removed for better visualization'

    return data


def tag_metadata_url(_url):
    """
    Gets all the tags from the newspaper library
    :param _input: url not containing a pdf
    :return:
    """

    from newspaper import Article, Config

    configuration = Config()
    configuration.request_timeout = 15  # default = 7
    configuration.keep_article_html = True

    article = Article(_url, config=configuration)

    article.download()
    article.html
    article.parse()
    article.nlp()

    data = {'url': _url,
            'publish_date': str(article.publish_date),
            'meta_lang': article.meta_lang,
            'meta_keywords': article.meta_keywords,
            'topics': article.meta_data.get('TOPICS', ''),
            'language': article.meta_data.get('LANGUAGE', ''),
            'publication_type': article.meta_data.get('PUBLICATION_TYPE', ''),
            'text': article.text,
            'full_text': article.title + " " + str(article.text),
            'article_html': article.article_html,
            'authors': article.authors,
            'title': article.title,
            'tags': list(article.tags),
            'keywords': article.keywords,
            'summary': article.summary,
            'top_image': article.top_image}

    if article.article_html == '':
        data['article_html'] = article.html  # takes all the html of the page

    import html2text  # Other libraries are tomd and pandoc
    data['body_markdown'] = html2text.html2text(str(data['article_html']))

    return data


def tag_metadata_url_pdf(_urlpdf):
    """
    Gets all the tags from the newspaper library
    :param _urlpdf: url containing a pdf
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux

    from newspaper import Article, Config

    configuration = Config()
    configuration.request_timeout = 15  # default = 7
    configuration.keep_article_html = True

    article = Article(_urlpdf, config=configuration)

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

    article.download()
    article.html

    article.parse()
    article.nlp()

    data = {'url': _urlpdf,
            'publish_date': str(article.publish_date),
            'meta_lang': article.meta_lang,
            'meta_keywords': article.meta_keywords,
            'topics': article.meta_data.get('TOPICS', ''),
            'language': article.meta_data.get('LANGUAGE', ''),
            'publication_type': article.meta_data.get('PUBLICATION_TYPE', ''),
            'text': article.text,
            'full_text': article.title + " " + str(article.text),
            'article_html': article.article_html,
            'authors': article.authors,
            'title': article.title,
            'tags': list(article.tags),
            'keywords': article.keywords,
            'summary': article.summary,
            'top_image': article.top_image}

    if article.article_html == '':
        data['article_html'] = article.html  # takes all the html of the page

    import html2text  # Other libraries are tomd and pandoc
    data['body_markdown'] = html2text.html2text(str(data['article_html']))

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
    import json

    places = GeoText(_dict_in['full_text'])
    _dict_in['cities'] = places.cities
    _dict_in['nationalities'] = places.nationalities
    _dict_in['countries_iso2'] = json.dumps(places.country_mentions)

    _dict_in['primary_country'] = ""
    if len(places.country_mentions) > 0:
        prim_country_iso2 = list(places.country_mentions)[0]
        if prim_country_iso2 == 'UK': # GeoText gets UK as iso2 UK and pycountry get it as GB .
        # Issue: https://github.com/elyase/geotext/issues/15
            prim_country_iso2 = 'GB'
        try: # In case that the FIPS code is not recognized by pycountry
            country = pycountry.countries.get(alpha_2=prim_country_iso2)
            _dict_in['primary_country'] = [country.name, country.alpha_3]
            _dict_in['countries'] = []
        except Exception as e:
            _dict_in['primary_country'] = []
            _dict_in['countries'] = []
            _dict_in['error'] = 'The primary country identified ' + prim_country_iso2 + \
                                ' has a FIPS code not matching any ISO2'
    while len(places.country_mentions) > 0:
        c = places.country_mentions.popitem(last=False)
        iso2 = c[0]
        if iso2 == 'UK':
            iso2 = 'GB'
        try:  # In case that the FIPS code is not recognized by pycountry
            country = pycountry.countries.get(alpha_2=iso2)
            _dict_in['countries'].append((country.name, country.alpha_3, c[1]))
        except Exception as e:
            _dict_in['countries'] = []
            _dict_in['error'] = 'A country identified ' + iso2 + ' has a FIPS code not matching any ISO2'
