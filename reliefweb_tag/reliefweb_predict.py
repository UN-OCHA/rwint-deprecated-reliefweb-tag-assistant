def url_to_tagged_json(model, url, threshold=0.5, diff_terms=0.1):
    """
    Main method to tag a URL
    :param model:
    :param url:
    :param threshold:
    :param diff_terms:
    :return:
    """

    import json

    sample_dict = tag_metadata_from_url(url)
    tag_language_langdetect(sample_dict)
    tag_country_basic(sample_dict)
    tag_language(model['language'], sample_dict)
    tag_theme(model['theme'] , sample_dict, threshold, diff_terms)

    return json.dumps(sample_dict, indent=4)


def tag_metadata_from_url(url):
    """
    Gets all the tags from the newspaper library
    :param url:
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux

    from newspaper import Article, Config

    configuration = Config()
    configuration.request_timeout = 15  # default = 7
    configuration.keep_article_html = True

    article = Article("")
    # if URL IS PDF or any binary then
    if url.lower()[-4:] in [".pdf"]:
        article = Article(url, config=configuration)
        pdf = reliefweb_tag_aux.get_pdf_url(url)

        pdf_text = ' '.join(pdf)
        article.set_text(pdf_text)
        article.set_article_html(pdf_text)
        article.set_html(pdf_text)
        article.title = pdf.metadata[0].get('Title',
                                            '')  # set title fills the field with Configuration when title empty
        article.set_authors([pdf.metadata[0].get('Author', '')])
        article.publish_date = pdf.metadata[0].get('CreationDate', '')

    else:
        # if it is not pdf
        article = Article(url, config=configuration)
        article.download()

    article.html
    article.parse()
    article.nlp()

    data = {}
    data['publish_date'] = str(article.publish_date)
    data['meta_lang'] = article.meta_lang
    data['meta_keywords'] = article.meta_keywords
    data['topics'] = article.meta_data.get('TOPICS', '')
    data['language'] = article.meta_data.get('LANGUAGE', '')
    data['publication_type'] = article.meta_data.get('PUBLICATION_TYPE', '')
    data['text'] = article.text
    data['full_text'] = article.title + " " + article.text
    # data['html'] = article.html # html of the whole page
    data['article_html'] = article.article_html
    data['authors'] = article.authors
    data['title'] = article.title
    data['tags'] = list(article.tags)
    data['keywords'] = article.keywords
    data['summary'] = article.summary
    data['top_image'] = article.top_image

    return data


def tag_theme(model, dict, threshold, diff_terms):
    """
    Creates the 'theme' value on the dictionary based on the theme neural model
    :param model:
    :param dict:
    :param threshold:
    :param diff_terms:
    :return:
    """

    text = dict['full_text']
    predicted_value = model.predict_nonlanguage_text(sample=text,
                                                     vocabulary_name='theme',
                                                     threshold=threshold,
                                                     diff_terms=diff_terms)
    dict['theme'] = predicted_value
    return dict


def tag_language(model, dict):
    """
    Creates the 'predicted_lang' value on the dictionary based on the language neural model
    :param model:
    :param dict:
    :return:
    """

    predicted_value = model.predict_language(dict['full_text'])
    dict['predicted_lang'] = predicted_value
    return dict


def tag_language_langdetect(dict):
    """
    Creates the 'langdetect_language' value on the dictionary based on the theme neural model
    :param dict:
    :return:
    """

    from reliefweb_tag import reliefweb_tag_aux
    dict['langdetect_language'] = reliefweb_tag_aux.detect_language(dict['full_text'])
    return dict


def tag_country_basic(dict):
    """
    Creates the 'countries', 'primary_country', 'countries_iso2', 'cities', 'nationalities' value on the dictionary
    using the GeoText module

    TODO: Geotagging with coordinates
    from geopy.geocoders import Nominatim
    geolocator = Nominatim()
    location = geolocator.geocode("175 5th Avenue NYC")
    print(location.address, location.latitude, location.longitude))

    :param dict:
    :return:
    """

    from geotext import GeoText
    import pycountry

    places = GeoText(dict['full_text'])
    dict['cities'] = places.cities
    dict['nationalities'] = places.nationalities
    dict['countries_iso2'] = places.country_mentions

    dict['primary_country'] = ""
    if len(places.country_mentions) > 0:
        country = pycountry.countries.get(alpha_2=list(places.country_mentions)[0])
        dict['primary_country'] = [country.name, list(places.country_mentions)[0]]

    dict['countries'] = []
    while len(places.country_mentions) > 0:
        c = places.country_mentions.popitem(last=False)
        iso2 = c[0]
        if iso2 == 'UK':
            iso2 = 'GB'
        country = pycountry.countries.get(alpha_2=iso2)
        dict['countries'].append((country.name, iso2, c[1]))
