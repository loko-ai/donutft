import os
import json
from pathlib import Path
import shutil


def md():
    # define paths
    base_path = Path("/home/fulvio/projects/donutft/ftdata")
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    # define metadata list
    metadata_list = []

    # parse metadata
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            # load json file
            data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(data)
            # add to metadata list if image exists
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append({"text": text, "file_name": f"{file_name.stem}.jpg"})
                # delete json file

    # write jsonline file
    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    # remove old meta data
    # shutil.rmtree(metadata_path)


import os
import json
from pathlib import Path
import shutil
from datasets import load_dataset

# define paths
base_path = Path("/home/fulvio/projects/donutft/ftdata")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")

# Load dataset
dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

print(f"Dataset has {len(dataset)} images")
print(f"Dataset features are: {dataset.features.keys()}")

new_special_tokens = []  # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>"  # eos token of tokenizer


def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                        fr"<s_{k}>"
                        + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}


proc_dataset = dataset.map(preprocess_documents_for_donut)

print(f"Sample: {proc_dataset[10]['text']}")
print(f"New special tokens: {new_special_tokens + [task_start_token] + [eos_token]}")

from transformers import DonutProcessor

# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [720, 960]  # should be (width, height)
processor.feature_extractor.do_align_long_axis = False


def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}


# need at least 32-64GB of RAM to run this
processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image", "text"])
processed_dataset = processed_dataset.train_test_split(test_size=0.1)

from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"New embedding size: {new_emb}")
# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1]  # (height, width)
model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# hyperparameters used for multiple args
hf_repository_id = "donut-base-sroie"

# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
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
    train_dataset=processed_dataset["train"],
)
trainer.train()
