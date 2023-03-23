import json
from pathlib import Path

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from donutft.business.preprocessing.ds import DonutPreprocessor
from donutft.business.preprocessing.processor import get_donut_processor
from donutft.business.preprocessing.tokens import JSON2Token, Transform
from donutft.model.donutmodel import get_donut_model

base_path = Path("/home/fulvio/projects/donutft/ftdata/img")
ds = datasets.load_dataset("imagefolder", data_dir=base_path, split="train[0:50]")

print(ds[0])
j2t = JSON2Token()

nt = set()
for el in ds:
    keys = list(json.loads(el['text']).keys())
    for k in keys:
        nt.add(f"<s_{k}>")
        nt.add(f"</s_{k}>")

start_token = "<s>"
end_token = "</s>"

print(list(nt))
pre = DonutPreprocessor(start_token, end_token, j2t)
processor = get_donut_processor(list(nt))

model = get_donut_model(processor)
t = Transform(processor)

train = [t(pre(x)) for x in tqdm(ds)]

training_args = Seq2SeqTrainingArguments(
    output_dir="mysroie",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters

    push_to_hub=False,
)

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
)
trainer.train()
processor.save_pretrained("./sroie2")
model.save_pretrained("./sroie2")
