from flask import Flask, request
from cassandra.cluster import Cluster

cluster = Cluster(contact_points=['172.17.0.2'], port=9042)
session = cluster.connect()
app = Flask(__name__)

@app.route('/')
def hello():
        name = request.args.get("name", "World")
        return('<h1>Hello, {}!</h1>'.format(name))

@app.route('/pokemon/<name>')
def profile(name):
        rows = session.execute("""SELECT * FROM pokemon.stats WHERE name='{}'""".f$
        for row in rows:
                return('<h3>{} has {} attack!</h3>'.format(name, row.attack))

        return('<h3>Does not exist!!!</h3>')

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=80)
