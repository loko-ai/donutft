import sys

from PIL import Image

# import torchvision.transforms as T

# import torchvision.transforms.functional as F

# tensor_transformer = T.ToPILImage()



class JSON2Token:
    def __init__(self):

        self.new_special_tokens = []

    def __call__(self, obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
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
                        self.new_special_tokens.append(
                            fr"<s_{k}>") if fr"<s_{k}>" not in self.new_special_tokens else None
                        self.new_special_tokens.append(
                            fr"</s_{k}>") if fr"</s_{k}>" not in self.new_special_tokens else None
                    output += (
                            fr"<s_{k}>"
                            + self(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            # excluded special tokens for now
            obj = str(obj)
            if f"<{obj}/>" in self.new_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj


class Transform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, sample, max_length=512, ignore_id=-100):
        # create tensor from image
        try:
            pixel_values = self.processor(
                sample["image"], do_resize=True, resample=4, random_padding=True, return_tensors="pt"
            ).pixel_values.squeeze()
            # img = F.to_pil_image(pixel_values.to("cpu"))
            # img_name = "img_postproc/200_dpi_resampling4_align/img_" + str(img_id) + ".png"
            # img.save(img_name)
            # sys.exit(0)

        except Exception as e:
            print(sample)
            print(f"Error: {e}")
            return {}

        # tokenize document
        input_ids = self.processor.tokenizer(
            sample["text"],
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",

        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
        return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

