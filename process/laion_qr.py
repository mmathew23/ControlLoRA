import os
import cv2
import torch
import torchvision.transforms.functional as TF
import random
import numpy as np

from PIL import Image
from process import base
from datasets.load import load_dataset


def gaussian_kernel(size: int, sigma: float):
    # Create a coordinate grid
    x = torch.linspace(-(size // 2), size // 2, steps=size)

    # Compute 1D Gaussian distribution
    gauss = torch.exp(-(x**2) / (2 * sigma**2))

    # Normalize to ensure kernel sums to 1
    gauss /= gauss.sum()

    # Compute the outer product to get the 2D kernel
    return gauss[:, None] * gauss[None, :]


def convolve_gaussian(img, size, sigma=1, padding=0, stride=1):
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)[None]
    weights = gaussian_kernel(size, sigma).to(img.device)
    if stride == 0:
        stride = size
        assert img.shape[1] % size == 0 and img.shape[2] % size == 0
    return torch.nn.functional.conv2d(img, weights[None, None, :, :], padding=padding, stride=stride)

# function to quantize an image


def quantize(img: Image.Image, res: int = 512, module_size: int = 10, border: int = 4, **kwargs):
    # first convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # resize to res
    if img.size[0] != res:
        img = img.resize((res, res), resample=Image.BILINEAR)

    # convert to torch tensor
    img = TF.to_tensor(img)

    # check if image is divisible by module_size
    # if img.shape[1] % module_size != 0 or img.shape[2] % module_size != 0:
    #     raise ValueError("Image dimensions must be divisible by module_size")
    
    # # create gaussian kernel according to module_size
    # gaussian_img = convolve_gaussian(img[None], module_size, sigma=1.5)
    # # threshold image
    # # img = gaussian_img > threshold
    # min_val = gaussian_img.min()
    # max_val = gaussian_img.max()
    # mean = gaussian_img.mean()
    # offset = kwargs.get('offset', (max_val - min_val) * 0.075)
    # lower_threshold = mean - offset
    # upper_threshold = mean + offset

    # # set lower threshold to 0
    # gaussian_img[gaussian_img < lower_threshold] = 0
    # # set upper threshold to 1
    # gaussian_img[gaussian_img > upper_threshold] = 1
    # # set in between to random uniform gray values between 0.4 and 0.6
    # gaussian_img[(gaussian_img > lower_threshold) & (gaussian_img < upper_threshold)] = torch.rand_like(gaussian_img[(gaussian_img > lower_threshold) & (gaussian_img < upper_threshold)]) * 0.2 + 0.4
    # img = gaussian_img

    # # create white border around image without padding extra pixels
    # # img is a torch tensor in b, c, h, w format
    # # no padding
    # if border != 0:
    #     img[:,:,:border, :] = 1
    #     img[:,:,-border:, :] = 1
    #     img[:,:,:,:border] = 1
    #     img[:,:,:,-border:] = 1

    # # apply contrast
    # # img = TF.adjust_contrast(gaussian_img, 5)

    # # resize back to res using nearest neighbor
    # img = torch.nn.functional.interpolate(img.float(), size=(res, res), mode='nearest')

    # return img[0]

    # # find the 25th percentile of the image and set it to lower threshold
    # lower_threshold = torch.quantile(img, 0.25)

    # # find the 75th percentile of the image and set it to upper threshold
    # upper_threshold = torch.quantile(img, 0.75)

    # gray_img = torch.ones_like(img) * 0.5
    # # set lower threshold to 0
    # gray_img[img <= lower_threshold] = 0
    # # set upper threshold to 1
    # gray_img[img >= upper_threshold] = 1
    # return gray_img

    gaussian_img = convolve_gaussian(img, module_size, sigma=3.5, padding=module_size//2, stride=1)
    contrast = kwargs.get('contrast', 5.0)
    gaussian_img = TF.adjust_contrast(gaussian_img, contrast)
    return gaussian_img[0]


class Dataset(base.Dataset):
    def __init__(self, tokenizer, resolution=512, use_crop=True, **kwargs):
        self.tokenizer = tokenizer

        self.dataset_type = kwargs.get('dataset_type', 'train')
        self.dataset = load_dataset('/media/mathew/ExternalDrive/datasets/QR', self.dataset_type)['train']
        self.size = resolution
        self.use_crop = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img_path = item['image']
        img = Image.open(img_path).convert('RGB')

        if self.use_crop:
            w, h = img.size
            x0 = torch.randint(0, w - self.size, (1, )).item() if w > self.size else 0
            y0 = torch.randint(0, h - self.size, (1, )).item() if h > self.size else 0
            x1 = x0 + self.size
            y1 = y0 + self.size
            img = img.crop((x0,y0,x1,y1))
            
        w, h = img.size
        
        if self.dataset_type == 'validation':
            module_size = 11
            border = 0
            guide = 2*quantize(img, res=self.size, module_size=module_size, border=border, offset=0, contrast=1.0)-1.0
        else:
            # border is random int between 3 and 8
            # border = random.randint(3, 6)
            border = 0
        
            # module size is random choice of (4, 8, 16)
            # module_size = random.choice([1, 2, 4, 8, 16])
            module_size = random.choice([5, 9, 13, 17])
            guide = 2*quantize(img, res=self.size, module_size=module_size, border=border)-1.0
        # guide is 1, h, w tensor and expand to 3, h, w
        guide = guide.expand(3, h, w) 



        img = 2*TF.to_tensor(img)-1.0
        input_ids = self.tokenizer({"text": [item["text"]]})[0]

        return { "pixel_values": img, "guide_values": guide, "input_ids": input_ids, "text": item["text"] }

    @staticmethod
    def control_channel():
        return 3

    @staticmethod
    def cat_input(image: Image.Image, target: torch.Tensor, guide: torch.Tensor):
        target = np.uint8(((target + 1) * 127.5)[0].permute(1,2,0).cpu().numpy().clip(0,255))
        guide = np.uint8(((guide + 1) * 127.5)[0].permute(1,2,0).cpu().numpy().clip(0,255))
        target = Image.fromarray(target).convert('RGB').resize(image.size)
        guide = Image.fromarray(guide).convert('RGB').resize(image.size)
        image_cat = Image.new('RGB', (image.size[0]*3,image.size[1]), (0,0,0))
        image_cat.paste(target,(0,0))
        image_cat.paste(guide,(image.size[0],0))
        image_cat.paste(image,(image.size[0]*2, 0))

        return image_cat

Dataset.register_cls('laion_qr')
