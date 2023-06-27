import io
import json
import os
import sys
from pathlib import Path

import fitz
from PIL import Image
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from donutft.business.preprocessing.processor import get_donut_processor
from donutft.business.preprocessing.tokens import JSON2Token, Transform
from donutft.model.donutmodel import get_donut_model

source = Path(os.environ.get('SOURCE'))

js2t = JSON2Token()

start_token = "<s>"
end_token = "</s>"


def get_dataset(source: Path):
    with open(source/'annotazioni.json') as f:
        dataset = json.load(f)
    for el in dataset:
        fname = el['fname']
        # with open(source/f'preprocessed/fattura/{fname}.txt') as f:
        #     text = f.read()
        # el['text'] = text
        doc = fitz.open(source/f'fatture/{fname}.pdf')
        pix = doc[0].get_pixmap(dpi=100)
        data = pix.pil_tobytes("JPEG")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        md = el.copy()
        del md['fname']
        del md['bollo']
        del md['numero_servizi']
        if 'controllare' in md:
            del md['controllare']
        md['prestazioni'] = [{k: v for k, v in x.items() if v} for x in md['prestazioni']]
        md['prestazioni'] = [x for x in md['prestazioni'] if x]
        md = {k:v for k,v in md.items() if v}
        yield dict(image=img, text=start_token + js2t(md) + end_token)


ds = list(get_dataset(source))

# sys.exit(0)

processor = get_donut_processor(list(js2t.new_special_tokens))

model = get_donut_model(processor)
t = Transform(processor)

train = [t(x) for x in ds]
# print(train)
hf_repository_id = "fatture"

training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    # fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    no_cuda=True,
    # push to hub parameters

    push_to_hub=False,
)

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
)

# sys.exit(0)
trainer.train()

processor.save_pretrained("./fatture_model")
model.save_pretrained("./fatture_model")