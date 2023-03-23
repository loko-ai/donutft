from transformers import DonutProcessor
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

img = Image.open("/home/fulvio/projects/donutft/data/scontrini-approvati-7.jpg")
img = img.convert("RGB")

out = processor(img)

print(processor.tokenizer("<s>Ciao io sono Fulvio <a>hhhs</a>"))
print(processor.tokenizer.pad_token_id)
