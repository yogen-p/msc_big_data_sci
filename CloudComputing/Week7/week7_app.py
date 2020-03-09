from flask import Flask, render_template, request, jsonify
import json
import requests
import requests_cache
requests_cache.install_cache('crime_api_cache', backend='sqlite', expire_after=36000)
app = Flask(__name__)
crime_url_template = 'https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lng}&date={date}'
categories_url_template = 'https://data.police.uk/api/crime-categories?date={date}'
@app.route('/crimestat', methods=['GET'])
def crimechart():
    my_latitude = request.args.get('lat','51.52369')
    my_longitude = request.args.get('lng','-0.0395857')
    my_date = request.args.get('date','2018-11')
    crime_url = crime_url_template.format(lat = my_latitude, lng = my_longitude, data = m$
    resp = requests.get(crime_url)
    if resp.ok:
        return jsonify(resp.json())
    else:
        print(resp.reason)

@app.route('/crimestat/all_cats/date=<date>', methods=['GET'])
def categories(date):
    my_latitude = request.args.get('lat','51.52369')
    my_longitude = request.args.get('lng','-0.0395857')
    my_date = date
    crime_url = crime_url_template.format(lat = my_latitude, lng = my_longitude, data = m$
    resp = requests.get(crime_url)
    if resp.ok:
        crimes = resp.json()
    else:
        print(resp.reason)

    categories_url = categories_url_template.format(date = my_date)
    resp = requests.get(categories_url)
    if resp.ok:
        cat_json = resp.json()
    else:
        print(resp.reason)

    cats = {cat["url"]: cat["name"] for cat in cat_json}

    stats = dict.fromkeys(cats.keys(), 0)
    stats.pop("all-crime")

    for crime in crimes:
        stats[crime["category"]] += 1

    return jsonify(stats)

if __name__=="__main__":
    app.run(host='0.0.0.0')

