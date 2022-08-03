from paddleocr import PaddleOCR,draw_ocr
import numpy as np
from PIL import Image, ImageDraw


OUT_PATH = "00000395.tif"  

ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.3, det_algorithm="DB") # need to run only once to download and load model into memory
result = ocr.ocr(OUT_PATH, cls=True) 
print(result)
