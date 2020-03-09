import requests
from pprint import pprint

url_temp = 'https://data.police.uk/api/crimes-street/all_crime?lat={lat}&lng={lng}&date={date}'

my_lat = '51.52369'
my_lng = '-0.0395857'
my_dat = '2018-11'

url = url_temp.format(lat=my_lat,
        lng=my_lng,
        date=my_dat)

resp = requests.get(url)
if resp.ok:
        crimes = resp.json()
else:
        print(resp.reason)

url_temp = 'https://data.police.uk/api/crime-categories?date={date}'

resp = requests.get(url_temp.format(date=my_dat))
if resp.ok:
    cat_json = resp.json()
else:
    print(resp.reason)

cats = {cat["url"]: cat["name"] for cat in cat_json}

stats = dict.fromkeys(cats.keys(), 0)
stats.pop("all-crime")

for crime in crimes:
    stats[crime["category"]] += 1

pprint(stats)

