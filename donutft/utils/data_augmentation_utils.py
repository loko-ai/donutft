import random
from PIL import Image, ImageOps, ImageEnhance


class ImageAugmenter:
    def __init__(self, img, fname):
        self.fname = fname
        self.img = img

    def _img_noising(self):
        img = self.img
        width, height = self.img.size
        for i in range(round(width * height / 5)):
            img.putpixel(
                (random.randint(0, width - 1), random.randint(0, height - 1)),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
        return img

    def _img_rotation(self,  degree):
        img = self.img.rotate(degree, Image.BICUBIC, expand=1)
        return img


    def _img_greyscale(self):
        img = ImageOps.grayscale(self.img)
        return self.img


    def _img_resize(self, upsize_percentage=0.15):
        width, height = self.img.size
        new_width = int(width + width * upsize_percentage)
        new_height = int(height + height * upsize_percentage)
        img = self.img.resize((new_height, new_width), Image.LANCZOS)
        return img

    def _img_brightner(self, brightness_factor=1.5):
        enhancer = ImageEnhance.Brightness(self.img)
        img = enhancer.enhance(brightness_factor)
        return img

    def _img_sharpner(self, sharpner_factor=1.3):
        enhancer = ImageEnhance.Sharpness(self.img)
        img = enhancer.enhance(sharpner_factor)
        return img

    def _img_contrast(self, contrast_factor=1.3):
        enhancer = ImageEnhance.Contrast(self.img)
        img = enhancer.enhance(contrast_factor)
        return img

    # def save_img_and_annotation(img, fname, augm_type):
    #     img.save(img, fname+"_"+augm_type)
    def __call__(self, *args, **kwargs):
        augmentend_img_list = []
        # img_contrasted = self.img_contrast(1.3)
        # img_sharped = self.img_sharpner(1.3)
        # img_brightned = self.img_brightner(1.3)
        # img_resize = self.img_resize(0.2)
        # img_rotated20 = self.img_rotation(20)
        # img_rotated5 = self.img_rotation(5)
        augmentend_img_list.append(self._img_contrast(1.3))
        augmentend_img_list.append(self._img_sharpner(1.3))
        augmentend_img_list.append(self._img_brightner(1.3))
        augmentend_img_list.append(self.img_brightner(0.8))
        augmentend_img_list.append(self._img_resize(0.2))
        augmentend_img_list.append(self._img_rotation(20))
        augmentend_img_list.append(self._img_rotation(5))
        return augmentend_img_list



