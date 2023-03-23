from transformers import DonutProcessor, DonutImageProcessor


def get_donut_processor(tokens):
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.image_processor.size = [720, 960]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    processor.tokenizer.add_special_tokens(dict(additional_special_tokens=tokens))
    return processor
