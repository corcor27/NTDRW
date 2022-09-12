from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import cv2
import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFont
import PIL
import os
from pdf2image import pdfinfo_from_path,convert_from_path

from NTDRW_CLASS import Document_Analysis
from models import dice_loss, selective_unet, dice_coeff
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from paddleocr import PaddleOCR,draw_ocr
import wget


BASE_FOLDER = "PRIMA_IMAGES"
IMG_LIST = os.listdir(BASE_FOLDER)
print(len(IMG_LIST))


IMG_WIEGHTS = "https://drive.google.com/file/d/1gBKFDy7Uogj0xqtKtkKBk7Ai7JgsEydX/view?usp=sharing"

TEXT_WIEGHTS = "https://drive.google.com/file/d/1oZ1tJcDy9ki7owKA0gv7aMsTH83j6gId/view?usp=sharing"

if os.path.exists("SEGMENTATION_ML_0.h5") == False:
   IMG = wget.download(IMG_WIEGHTS, out="SEGMENTATION_ML_0.h5")
if os.path.exists("SEGMENTATION_TEXT_0.h5")==False:
   TEXT = wget.download(TEXT_WIEGHTS, out="SEGMENTATION_TEXT_0.h5")

for IMG_PATH in range(0, 1):
    #start = Document_Analysis("1041245.jpg", True, "iiif_saved2")
    start = Document_Analysis(IMG_LIST[IMG_PATH], True, BASE_FOLDER)
    

#print(result)
#print(boxes)

