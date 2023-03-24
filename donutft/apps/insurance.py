import itertools
import json
from pathlib import Path

import fitz

p = Path("/home/fulvio/projects/donutft/data/SAMPLE_DOC_FULVIO/tipo_2")

out = Path("./insurance_data")
out.mkdir(exist_ok=True)
md = dict(protocollo="", richiedente="", cognome="", nome="", codice_fiscale="")
for el in p.glob("*.pdf"):
    print(el)
    doc = fitz.open(el)

    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        text = page.get_text()
        name = f"{el.stem}_page_{i}"
        pix.save(out / f"{name}.jpg")
        with open(out / f"{name}.json", "w") as o:
            json.dump(dict(md, text=text), o)
        print(pix)
