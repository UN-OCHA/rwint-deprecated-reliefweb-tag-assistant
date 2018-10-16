def compare_results(_scope, _sample_dict):
    url = _sample_dict["url"]

    if not url.startswith("https://reliefweb.int"):
        _sample_dict['error'] = "The URL is not ReliefWeb's, so it cannot be evaluated"
        raise Exception("The URL is not ReliefWeb's, so it cannot be evaluated")

    import urllib.request

    if _scope == "job":
        rw_api_call_url = "https://api.reliefweb.int/v1/jobs?appname=tests&" \
                          "filter[field]=url_alias&" \
                          "fields[include][]=city&" \
                          "fields[include][]=country&" \
                          "fields[include][]=experience&" \
                          "fields[include][]=career_categories&" \
                          "fields[include][]=type&" \
                          "fields[include][]=theme&" \
                          "filter[value]=" + url
    else:
        _sample_dict[
            'error'] = "RW Tag Assistant can only compare jobs by now. Reports are not supported. Change the scope parameter"
        raise Exception(
            "RW Tag Assistant can only compare jobs by now. Reports are not supported. Change the scope parameter")

    print("Evaluating " + url)
    req = urllib.request.Request(rw_api_call_url)
    try:
        with urllib.request.urlopen(req) as response:
            json_bytes = response.read()
        import json
        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        my_json = json_bytes.decode('utf8')
        # Load the JSON to a Python list & dump it back out as formatted JSON
        data = json.loads(my_json)
        rw_fields = data['data'][0]['fields']

        if _scope == "job":
            evaluation = ""
            evaluation = evaluation + compare_values("city", rw_fields["city"], _sample_dict["city"])
            evaluation = evaluation + compare_values("country", rw_fields["country"], _sample_dict["primary-country"])
            evaluation = evaluation + compare_values("experience", rw_fields["experience"],
                                                     _sample_dict["job-experience"])
            evaluation = evaluation + compare_values("career_categories", rw_fields["career_categories"],
                                                     _sample_dict["job-category"])
            evaluation = evaluation + compare_values("type", rw_fields["type"], _sample_dict["job-type"])
            evaluation = evaluation + compare_values("theme", rw_fields["theme"], _sample_dict["job-theme"])
            # TODO: Theme is multiple valued. Algorithm for comparing
            # TODO: _sample-dict['xx'] are multiple items, take the first one
            # TODO: Take into account the probability
            _sample_dict["evaluation"] = evaluation


    except Exception as e:
        print("ERROR: While calling " + rw_api_call_url)
        _sample_dict['error'] = str(e)
        return _sample_dict

    s = json.dumps(data, indent=4, sort_keys=True)  # for debugging and printing
    _sample_dict["rw_api"] = s
    return _sample_dict


def compare_values(field_name, value_api, value_predicted):
    if value_api == value_predicted:
        return "\\t" + field_name + ": MATCH - " + value_predicted
    else:
        return "\\t" + field_name + ": NO MATCH - RW: " + value_api + " / PREDICTED: " + value_predicted
