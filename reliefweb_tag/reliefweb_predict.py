def predict(_models, _input, _scope):
    """
    Main method to tag a URL or text, return a JSON with all the fields populated and predicted
    :param models: array of machine learning models
    :param _input: url format or free text
    :param scope: scope to match for each ML model to tag on that vocabulary
    :return:
    """

    import json

    #try:
    sample_dict = tag_metadata(_input)
    tag_language_langdetect(sample_dict)
    tag_country_basic(sample_dict)

    for each in _models.keys():
        if (_models[each].config['scope'] == _scope) or (_models[each].config['scope'] == "all"):
            tag_machine_learning_model(_models[each], sample_dict)



    #except Exception as e:
    #    sample_dict = {}
    #    sample_dict['error'] = str(e)
    #    sample_dict['full_text'] = ''

    return json.dumps(sample_dict, indent=4)


def tag_metadata(input):
    """
    Gets all the tags from the newspaper library
    :param input: can be a url or a text string
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux

    from newspaper import Article, Config

    configuration = Config()
    configuration.request_timeout = 15  # default = 7
    configuration.keep_article_html = True

    isurl = (input.lower()[:7] == 'http://') or (input.lower()[:8] == 'https://')

    if isurl:
        article = Article(input, config=configuration)
    else:
        article = Article("")

    # if URL IS PDF or any binary then
    if input.lower()[-4:] in [".pdf"]:
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

    if (not isurl):
        article.download(input)
        article.set_text(input)
        article.set_article_html(input)
        article.set_html(input)
        article.title = input
    else:
        # if it is not pdf
        article.download()
        article.html

    try:
        article.parse()
        article.nlp()
    except Exception as e:
        raise Exception(e)

    data = {'publish_date': str(article.publish_date),
            'meta_lang': article.meta_lang,
            'meta_keywords': article.meta_keywords,
            'topics': article.meta_data.get('TOPICS', ''),
            'language': article.meta_data.get('LANGUAGE', ''),
            'publication_type': article.meta_data.get('PUBLICATION_TYPE', ''),
            'text': article.text,
            'full_text': article.title + " " + article.text,
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
    data['body_markdown'] = html2text.html2text(data['article_html'])

    return data


def tag_machine_learning_model(model, dict_in):
    """
    Creates the 'theme' value on the dictionary based on the theme neural model
    :param model:
    :param dict_in:
    :param threshold:
    :param diff_terms:
    :return:
    """

    text = dict_in['full_text']
    predicted_value = model.predict_nonlanguage_text(sample=text)
    dict_in[model.vocabulary_name] = predicted_value
    return dict_in


def tag_language_langdetect(dict_in):
    """
    Creates the 'langdetect_language' value on the dictionary based on the theme neural model
    :param dict_in:
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux
    dict_in['langdetect_language'] = reliefweb_tag_aux.detect_language(dict_in['full_text'])
    return dict_in


def tag_country_basic(dict_in):
    """
    Creates the 'countries', 'primary_country', 'countries_iso2', 'cities', 'nationalities' value on the dictionary
    using the GeoText module

    TODO: Geotagging with coordinates
    from geopy.geocoders import Nominatim
    geolocator = Nominatim()
    location = geolocator.geocode("175 5th Avenue NYC")
    print(location.address, location.latitude, location.longitude))

    :param dict_in:
    :return:
    """

    from geotext import GeoText
    import pycountry

    places = GeoText(dict_in['full_text'])
    dict_in['cities'] = places.cities
    dict_in['nationalities'] = places.nationalities
    dict_in['countries_iso2'] = places.country_mentions

    dict_in['primary_country'] = ""
    if len(places.country_mentions) > 0:
        country = pycountry.countries.get(alpha_2=list(places.country_mentions)[0])
        dict_in['primary_country'] = [country.name, list(places.country_mentions)[0]]
        dict_in['countries'] = []
    while len(places.country_mentions) > 0:
        c = places.country_mentions.popitem(last=False)
        iso2 = c[0]
        if iso2 == 'UK':
            iso2 = 'GB'
        country = pycountry.countries.get(alpha_2=iso2)
        dict_in['countries'].append((country.name, iso2, c[1]))
