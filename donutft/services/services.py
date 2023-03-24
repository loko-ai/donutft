import json
from pathlib import Path

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

app = Flask("ann")

p = Path("../apps/insurance_data")

md = dict(protocollo="", richiedente="", cognome="", nome="", codice_fiscale="")


@app.get("/docs")
def get_docs():
    return jsonify([x.stem for x in p.glob("*.jpg")])


@app.get("/docs/<id>")
def get_doc(id):
    return send_file((p / (id + ".jpg")).resolve())


@app.get("/metadata/<id>")
def get_metadata(id):
    name = p / (id + ".json")
    if name.exists():
        return send_file(name)
    else:
        return jsonify(md)


@app.put("/metadata/<id>")
def put_metadata(id):
    data = request.json

    with open(p / (id + ".json"), "w") as o:
        json.dump(data, o)
    return jsonify("Ok")


CORS(app)

app.run("0.0.0.0", 8080, debug=True)
