import io
import re
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from donutft.business.preprocessing.tokens import JSON2Token

fname = '5811559'
source = Path("/home/cecilia/loko/data/previnet")
doc = fitz.open(source/f'fatture/{fname}.pdf')
pix = doc[0].get_pixmap(dpi=100)
data = pix.pil_tobytes("JPEG")
image = Image.open(io.BytesIO(data)).convert("RGB")

js2t = JSON2Token()

start_token = "<s>"
end_token = "</s>"

device = "cpu"


def get_donut_processor(tokens):
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.image_processor.size = [720, 960]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    processor.tokenizer.add_special_tokens(dict(additional_special_tokens=tokens))
    return processor

def get_donut_model(processor, max_length=512, start_token="<s>"):
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # Adjust our image size and output sequence lengths
    model.config.encoder.image_size = processor.feature_extractor.size[::-1]  # (height, width)
    model.config.decoder.max_length = max_length

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([start_token])[0]
    return model

processor = DonutProcessor.from_pretrained("philschmid/donut-base-sroie")

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

pixel_values = processor(image, random_padding=True, return_tensors="pt").pixel_values
# pixel_values = self.processor(
#     sample["image"], random_padding=True, return_tensors="pt"
# ).pixel_values.squeeze()

pixel_values = torch.tensor(pixel_values)#.unsqueeze(0)

decoder_input_ids = processor.tokenizer(
    "<s>",
    add_special_tokens=False,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
).input_ids

outputs = model.generate(
    pixel_values.to(device),
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
print(sequence)
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
print(sequence)
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
res = processor.token2json(sequence)
print(res)