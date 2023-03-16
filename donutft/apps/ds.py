from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from donutft.apps.dataset import DonutDataset
from donutft.apps.ptl import DonutModelPLModule
import pytorch_lightning as pl

dataset = load_dataset("naver-clova-ix/cord-v2")
print(dataset)

from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel

image_size = [1280, 960]
max_length = 768

# update image_size of the encoder
# during pre-training, a larger image size was used
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = image_size  # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]

processor.image_processor.size = image_size[::-1]  # should be (width, height)
processor.image_processor.do_align_long_axis = False

train_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
                             split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                             sort_json_key=False,  # cord dataset is preprocessed, so no need for this,
                             processor=processor,
                             model=model,
                             )

val_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
                           split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                           sort_json_key=False,  # cord dataset is preprocessed, so no need for this
                           processor=processor,
                           model=model,
                           )

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

config = {"max_epochs": 30,
          "val_check_interval": 0.2,  # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "num_training_samples_per_epoch": 400,
          "lr": 3e-5,
          "train_batch_sizes": [4],
          "val_batch_sizes": [1],
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 300,  # 800/8*30/10, 10%
          "result_path": "./result",
          "verbose": True,
          }

model_module = DonutModelPLModule(config, processor, model, max_length=max_length, tdl=train_dataloader,
                                  vdl=val_dataloader)

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=config.get("max_epochs"),
    val_check_interval=config.get("val_check_interval"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision=16,  # we'll use mixed precision
    num_sanity_val_steps=0,
    callbacks=[early_stop_callback],
)

trainer.fit(model_module)
