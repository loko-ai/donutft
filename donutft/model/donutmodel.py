from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig


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
