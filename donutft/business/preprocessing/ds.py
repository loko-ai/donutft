import json


class DonutPreprocessor:
    def __init__(self, task_start_token, eos_token, text_processor):
        self.task_start_token = task_start_token
        self.eos_token = eos_token
        self.text_processor = text_processor

    def __call__(self, sample):
        # create Donut-style input
        text = json.loads(sample["text"])
        d_doc = self.task_start_token + self.text_processor(text) + self.eos_token
        # convert all images to RGB
        image = sample["image"].convert('RGB')
        return {"image": image, "text": d_doc}
