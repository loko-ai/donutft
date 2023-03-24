import json
from pathlib import Path

from PIL import Image
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from donutft.business.preprocessing.processor import get_donut_processor
from donutft.business.preprocessing.tokens import JSON2Token, Transform
from donutft.model.donutmodel import get_donut_model

source = Path("./insurance_data")

js2t = JSON2Token()

start_token = "<s>"
end_token = "</s>"


def get_dataset(source: Path):
    for el in source.glob("*.jpg"):
        md = source / (el.stem + ".json")
        if md.exists():
            with md.open() as inp:
                d = json.load(inp)
                yield dict(image=Image.open(el).convert("RGB"), text=start_token + js2t(d) + end_token)


ds = list(get_dataset(source))

processor = get_donut_processor(list(js2t.new_special_tokens))

model = get_donut_model(processor)
t = Transform(processor)

train = [t(x) for x in ds]

hf_repository_id = "insurance_wd"

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
trainer.train()

processor.save_pretrained("./insurance_model")
model.save_pretrained("./insurance_model")
