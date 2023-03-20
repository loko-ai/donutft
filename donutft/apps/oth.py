from PIL import Image
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from donutft.apps.dataset import DonutDataset
from donutft.apps.ptl import DonutModelPLModule
import pytorch_lightning as pl
import torch
import re

from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel

image_size = [1280, 960]
max_length = 768

# update image_size of the encoder
# during pre-training, a larger image size was used
config = VisionEncoderDecoderConfig.from_pretrained("./mm")
config.encoder.image_size = image_size  # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

model.save_pretrained("./gg")
processor.save_pretrained("./gg")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids


def extract(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = model.generate(
        pixel_values.to("cuda"),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    print(sequence)
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    return processor.token2json(sequence)


img = Image.open("/home/fulvio/projects/donutft/data/scontrini-approvati-7.jpg")
img = img.convert("RGB")

print(extract(img))
