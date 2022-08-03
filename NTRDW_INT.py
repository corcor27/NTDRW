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

from NDTRW_CLASS2 import Document_Analysis
from models import dice_loss, selective_unet, dice_coeff
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from paddleocr import PaddleOCR,draw_ocr


BASE_FOLDER = "PRIMA_IMAGES"
IMG_LIST = os.listdir(BASE_FOLDER)
print(len(IMG_LIST))
for IMG_PATH in range(280, len(IMG_LIST)):
    #start = Document_Analysis("1041245.jpg", True, "iiif_saved2")
    start = Document_Analysis(IMG_LIST[IMG_PATH], True, BASE_FOLDER)
    

#print(result)
#print(boxes)

