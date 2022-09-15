from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import PIL
import os
from pdf2image import pdfinfo_from_path,convert_from_path
import pickle
import shutil
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor, SwinForImageClassification

import pdfplumber
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from models import dice_loss, selective_unet, dice_coeff
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from flair.data import Sentence
from flair.models import SequenceTagger
import ast
from transformers import pipeline
from transformers import DetrFeatureExtractor, DetrForSegmentation
import torch
import re
import sys


class Document_Analysis():
    def __init__(self, Doc_name, OLD, folder):
        self.folder = folder
        self.Doc_name = Doc_name
        self.OLD = OLD
        self.img_positions, self.word_positions, self.Text_dic, self.Img_dic, self.SECOND_NAME, self.IMAGE_DETAILS, self.IMG_PATH= self.Prepare_document()
        
        
  
# reading the data from the file
        #with open("Doc_analysis/00000087/Img_Details/text_details00000087.txt") as f:
            #data = f.read()
          
        #print("Data type before reconstruction : ", type(data))
              
        # reconstructing the data as a dictionary
        #self.Text_dic = ast.literal_eval(data)
        #print(self.Text_dic)
        #self.analysis_text = self.word_understanding(self.Text_dic)
        self.Text_Sum = self.Text_summary(self.Text_dic)
        #self.SEG = self.Image_segmentation()
        
        print(self.Text_Sum)
        
    def Image_analysis(self, POSITIONS, IMG_PATH):
        #IMG_GRAY = cv2.imread(IMG_PATH,0)
        IMG_COL = cv2.imread(IMG_PATH)
        
        text_key_base = {}
        for jj in range(0, POSITIONS.shape[0]):
            y, x, h, w = POSITIONS[jj,:]
            image = IMG_COL[y:y+h,x:x+w,:]
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            print(image.shape)
            if image.shape[0] >= 20 and image.shape[1] >= 20:
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                # model predicts one of the 1000 ImageNet classes
                predicted_class_idx = logits.argmax(-1).item()
                text_key_base["{}_{}_{}_{}".format(y, x, h, w)] = model.config.id2label[predicted_class_idx]
                #print("Predicted class:", model.config.id2label[predicted_class_idx])
        return text_key_base
    
    def Image_segmentation(self):
        IMG_COL = cv2.imread(self.IMG_PATH)
        if ".tif" in self.IMG_PATH:
            IMG_PATH2 = self.IMG_PATH.replace("tif", "png")
            cv2.imwrite(IMG_PATH2, IMG_COL)
            image = Image.open(IMG_PATH2).convert('RGB')
        if ".tif" not in self.IMG_PATH:
            image = Image.open(self.IMG_PATH)
        image2 = ImageDraw.Draw(image)
        text_key_base = {}
        feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-panoptic')
        model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-101-panoptic')
           
        inputs = feature_extractor(images=IMG_COL, return_tensors="pt")
        outputs = model(**inputs)
        # model predicts COCO classes, bounding boxes, and masks
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        text_size = 15
        font = ImageFont.truetype("arial.ttf", text_size)
        for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
            shape = [(xmin, ymin), (xmax, ymax)]
            cl = p.argmax()
            image2.rectangle(shape,  outline ="red")
            image2.text((xmin, ymin), "{}".format(model.config.id2label[cl.item()]), fill=(0, 0, 0), font=font)
        image.save(os.join.path(self.IMAGE_DETAILS, "base2"))
        return 0
            
            

       
    def Text_analysis(self, POSITIONS, IMG_PATH, File_folder):
        words_key_base = {}
        img = cv2.imread(IMG_PATH,0)
        img2 = cv2.imread(IMG_PATH)
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        if ".tif" in IMG_PATH:
            #IMG_PATH = IMG_PATH.replace("tif", "png")
            #deal with .tif being elsewhere in the filename
            IMG_PATH = re.sub(r'(.*)tif',r'\1png', IMG_PATH)
            cv2.imwrite(IMG_PATH, img2)
        image64 = Image.open(IMG_PATH).convert('RGB')
        image68 = ImageDraw.Draw(image64)
        IMAGE_OUT = os.path.join(File_folder, "Text_samples")
        if os.path.exists(IMAGE_OUT) == False:
            os.mkdir(IMAGE_OUT)
        OUT1 = os.path.join(IMAGE_OUT, "base.png")
        for jj in range(0, POSITIONS.shape[0]):
            
            
            yy, xx, h, w = POSITIONS[jj,:]
            OUT = os.path.join(IMAGE_OUT, "{}_{}_{}_{}.png".format(yy, xx, h, w))
            cv2.imwrite(OUT, img2[yy:yy+h, xx:xx+w, :])
            result = self.model_predict(OUT)
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            
            ret, base = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            words_key = []
            for kk in range(0, len(boxes)):
                y = []
                x = []
                for ii in range(0, len(boxes[kk])):
                    x.append(boxes[kk][ii][0])
                    y.append(boxes[kk][ii][1])
                y0, y1 = np.min(y), np.max(y)
                x0, x1 = np.min(x), np.max(x)
                shape = [(xx+x0, yy+y0), (xx+x1, yy+y1)]
                text_size = 40
                font = ImageFont.load_default()
                
                
                base = img2[int(y0+yy):int(y1+yy),int(x0+xx):int(x1+xx), :]
                OUT = os.path.join(IMAGE_OUT, "{}.png".format(kk))
                cv2.imwrite(OUT, base)
                pixel_values = processor(images=base, return_tensors="pt").pixel_values
                    
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                words_key.append(generated_text)
                image68.rectangle(shape,  outline ="red")
                image68.text((x0+xx, y0+yy-text_size), "{}".format(generated_text), fill=(0, 0, 255), font=font)
            words_key_base["{}_{}_{}_{}".format(yy, xx, h, w)] = words_key
            
        image64.save(OUT1)
        return words_key_base
    def word_understanding(self, TEXT_DIC):
        
        key = TEXT_DIC.keys()
        for opt in key:
            Section = TEXT_DIC[opt]
            for sen in Section:
                #tagger = SequenceTagger.load("flair/pos-english")
                tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                # make example sentence
                sentence = Sentence(sen)
                # predict NER tags
                tagger.predict(sentence)
                for entity in sentence.get_spans('ner'):
                    print(entity)
                    
    def Text_summary(self, TEXT_DIC):
        
        key = TEXT_DIC.keys()
        LIS = []
        for opt in key:
            if len(TEXT_DIC[opt]) > 1:
                Section = TEXT_DIC[opt]
                for sec in Section:
                    LIS.append(sec)
        print(LIS)
        Text = ''.join(LIS)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        OUTPUT = summarizer(Text, max_length=200, min_length=10, do_sample=False)
        OUT2 = os.path.join(self.IMAGE_DETAILS, "Summary" + self.SECOND_NAME + ".txt")
        print(OUT2)
        #analysis_text = self.word_understanding(Text_dic)
        # open file for writing
        f = open(OUT2,"w")
        f.write( str(OUTPUT) )
        f.close()
        return OUTPUT
            
        
        
    def Word_Extraction(self, IMG_PATH, ty):
        #IMG_PATH2 = IMG_PATH.replace("." + ty, " ")
        IMG_PATH2 = IMG_PATH + "_gray" + "." + ty
        print(IMG_PATH2)
        gray_img = cv2.imread(IMG_PATH, 0)
        color_img = cv2.imread(IMG_PATH)
        ret, gray_img2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(IMG_PATH2, gray_img2)
        result = self.model_predict(IMG_PATH2)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        ret, base = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        base2 = np.zeros((gray_img.shape))
        #words_key = {}
        for kk in range(0, len(boxes)):
            y = []
            x = []

            for ii in range(0, len(boxes[kk])):
                x.append(boxes[kk][ii][0])
                y.append(boxes[kk][ii][1])
            y0, y1 = np.min(y), np.max(y)
            x0, x1 = np.min(x), np.max(x)
            shape = [(x0, y0), (x1, y1)]
            
            hieght, width = int(round((y1 - y0)/2)), int(round((x1 - x0)/2))
            base2[int(y0):int(y1 + hieght), int(x0):int(x1)] = 255
            #words_key["{}_{}_{}_{}".format(y0,y1,x0,x1)] = txts[kk]
        IMG2 = cv2.resize(color_img, (512, 512),interpolation=cv2.INTER_CUBIC)
        IMG2 = IMG2/255
        model = selective_unet()
        model.load_weights("SEGMENTATION_TEXT_0.h5")
        IMG2 = np.expand_dims(IMG2, axis=0)

        pred_labels = model.predict(IMG2, batch_size=1)
        results_accuracy = (pred_labels >= 0.5).astype(np.uint8)
        results_accuracy = np.squeeze(results_accuracy)
        results_accuracy = cv2.resize(results_accuracy, (color_img.shape[1], color_img.shape[0]),interpolation=cv2.INTER_CUBIC)
        results_accuracy = results_accuracy*255

        result = self.model_predict(IMG_PATH)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        ret, base = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        base2 = np.zeros((gray_img.shape))
        #words_key = {}
        for kk in range(0, len(boxes)):
            y = []
            x = []

            for ii in range(0, len(boxes[kk])):
                x.append(boxes[kk][ii][0])
                y.append(boxes[kk][ii][1])
            y0, y1 = np.min(y), np.max(y)
            x0, x1 = np.min(x), np.max(x)
            shape = [(x0, y0), (x1, y1)]
            
            hieght, width = int(round((y1 - y0)/2)), int(round((x1 - x0)/2))
            base2[int(y0):int(y1), int(x0):int(x1)] = 255


        results_accuracy = base2 + results_accuracy
        results_accuracy = results_accuracy.astype(np.uint8)

        #base2 = base2.astype(np.uint8)
        ret, thresh = cv2.threshold(results_accuracy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        results = []
        img_positions = []
        base4 = np.zeros((gray_img.shape[0], gray_img.shape[1]))
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            base3 = gray_img[y:y+h,x:x+w]
            
            
            if w*h >= (gray_img.shape[0]*gray_img.shape[1])/100:
                base4[y:y+h,x:x+w] = 255
                img_positions.append(np.array([y, x, h, w]))
        img_positions = np.array(img_positions)

        return  img_positions, base4
    
    def Image_Extraction(self, IMG_PATH):

        IMG = cv2.imread(IMG_PATH)
        gray_img = cv2.imread(IMG_PATH, 0)
        IMG2 = cv2.resize(IMG, (512, 512),interpolation=cv2.INTER_CUBIC)
        IMG2 = IMG2/255
        model = selective_unet()
        model.load_weights("SEGMENTATION_ML_0.h5")
        IMG2 = np.expand_dims(IMG2, axis=0)

        pred_labels = model.predict(IMG2, batch_size=1)
        results_accuracy = (pred_labels >= 0.8).astype(np.uint8)
        results_accuracy = np.squeeze(results_accuracy)
        results_accuracy = cv2.resize(results_accuracy, (IMG.shape[1], IMG.shape[0]),interpolation=cv2.INTER_CUBIC)
        results_accuracy = results_accuracy*255
        
        
        ret, thresh = cv2.threshold(results_accuracy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        #ret, thresh = cv2.threshold(results_accuracy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imwrite("test2.png", thresh)
        results = []
        img_positions = []
        base3 = np.zeros((IMG.shape[0], IMG.shape[1]))
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            base2 = IMG[y:y+h,x:x+w]
            
            
            if w*h >= (IMG.shape[0]*IMG.shape[1])/100:
                img_positions.append(np.array([y, x, h, w]))
                base2 = cv2.resize(base2, (256,256),interpolation=cv2.INTER_CUBIC)
                base3[y:y+h,x:x+w] = 255
                results.append(base2)
        #cv2.imwrite("testing.png", base3)
        img_positions = np.array(img_positions)
        #results = np.array(results)
        return img_positions, base3
        
    
    def convert_pdf(self, PDF_PATH, OUT):
            
        info = pdfinfo_from_path(PDF_PATH, userpw=None, poppler_path=None)

        maxPages = info["Pages"]
        for page in range(1, maxPages+1, 10) : 
            convert_from_path(PDF_PATH, dpi=200, output_folder=OUT, first_page=page, last_page = min(page+10-1,maxPages), fmt="JPEG" )
        return 0

    def model_predict(self, IMG_PATH):
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5, det_algorithm="DB") 
        result = ocr.ocr(IMG_PATH, cls=True)
        return result
    def Prepare_document(self):
        
        img_types = ["png", "jpg", "tif"]
        Doc_convert = "Doc_analysis"
        if os.path.exists(Doc_convert) == False:
            os.mkdir(Doc_convert)
        
        
        if "pdf" in self.Doc_name:
            File_folder = os.path.join(Doc_convert, self.Doc_name.replace(".pdf", ""))
            if os.path.exists(File_folder) == False:
                os.mkdir(File_folder)
            IMAGE_SAVE = os.path.join(File_folder,  "Img_Files")
            if os.path.exists(IMAGE_SAVE) == False:
                os.mkdir(IMAGE_SAVE)
            IMAGE_DETAILS = os.path.join(File_folder,  "Img_Details")
            if os.path.exists(IMAGE_DETAILS) == False:
                os.mkdir(IMAGE_DETAILS)
            IMAGE_OUT = os.path.join(File_folder, "Img_Masks")
            if os.path.exists(IMAGE_OUT) == False:
                os.mkdir(IMAGE_OUT)
            if self.OLD == True:
                convert = self.convert_pdf(self.Doc_name, IMAGE_SAVE)
                IMG_LIST = os.listdir(IMAGE_SAVE)
    
                for item in IMG_LIST:
                    IMG_PATH = os.path.join(IMAGE_SAVE, item)
                    OUT_PATH1 = os.path.join(IMAGE_OUT, "Img_mask" + item)
                    OUT_PATH2 = os.path.join(IMAGE_OUT, "Text_mask" + item)
                    
                    #run = self.Extract_words(boxes, img)
                    img_positions, img_mask = self.Image_Extraction(IMG_PATH)
                    word_positions, word_mask = self.Word_Extraction(IMG_PATH, "jpg")
                    Text_dic = self.Text_analysis(word_positions, IMG_PATH)
                    print(img_positions)
                    #analysis_text = self.word_understanding(Text_dic)
                    cv2.imwrite(OUT_PATH1, img_mask)
                    cv2.imwrite(OUT_PATH2, word_mask)
                    Img_dic = self.Image_analysis(img_positions,IMG_PATH)
                    OUT1 = os.path.join(IMAGE_DETAILS, "text_details" + item.replace("jpg", "txt"))
                    OUT2 = os.path.join(IMAGE_DETAILS, "img_details" + item.replace("jpg", "txt"))
                    #analysis_text = self.word_understanding(Text_dic)
                    # open file for writing
                    f = open(OUT1,"w")
                    f.write( str(Text_dic) )
                    f.close()
                    g = open(OUT2,"w")
                    g.write( str(Img_dic) )
                    g.close()
                    
                    
                    
            else:
                pdf = pdfplumber.open(self.Doc_name)
                page = pdf.pages[0]
                text = page.extract_text()
                print(text)
                pdf.close()
                         
                    
        else:
            for ty in img_types:
                if ty in self.Doc_name:
                    SECOND_NAME = self.Doc_name.replace(ty, "")
                    #File_folder = os.path.join(Doc_convert, self.Doc_name.replace(ty, ""))
                    #keep the file extension in the dir name
                    File_folder = os.path.join(Doc_convert, self.Doc_name)
                    if os.path.exists(File_folder) == False:
                        print("creating dir",File_folder)
                        os.mkdir(File_folder)
                    IMAGE_SAVE = os.path.join(File_folder,  "Img_Files")
                    if os.path.exists(IMAGE_SAVE) == False:
                        os.mkdir(IMAGE_SAVE)
                    IMAGE_OUT = os.path.join(File_folder, "Img_Masks")
                    if os.path.exists(IMAGE_OUT) == False:
                        os.mkdir(IMAGE_OUT)
                    IMAGE_DETAILS = os.path.join(File_folder,  "Img_Details")
                    if os.path.exists(IMAGE_DETAILS) == False:
                        os.mkdir(IMAGE_DETAILS)
                    
                    IMG_PATH = os.path.join(IMAGE_SAVE, self.Doc_name)
                    shutil.copy("{}/{}".format(self.folder, self.Doc_name), IMG_PATH)
                    OUT_PATH1 = os.path.join(IMAGE_OUT, "Img_mask" + self.Doc_name)
                    OUT_PATH2 = os.path.join(IMAGE_OUT, "Text_mask" + self.Doc_name)
                    
                    img = cv2.imread(IMG_PATH)
                    img_positions, img_mask = self.Image_Extraction(IMG_PATH)
                    word_positions, word_mask = self.Word_Extraction(IMG_PATH, ty)
                    Text_dic = self.Text_analysis(word_positions, IMG_PATH, File_folder)
                    print(img_positions)
                    cv2.imwrite(OUT_PATH1.replace(ty, ".png"), img_mask)
                    cv2.imwrite(OUT_PATH2.replace(ty, ".png"), word_mask)
                    Img_dic = self.Image_analysis(img_positions,IMG_PATH)
                    OUT1 = os.path.join(IMAGE_DETAILS, "text_details" + self.Doc_name.replace(ty, "txt"))
                    OUT2 = os.path.join(IMAGE_DETAILS, "img_details" + self.Doc_name.replace(ty, "txt"))
                    
                    #analysis_text = self.word_understanding(Text_dic)
                    # open file for writing
                    f = open(OUT1,"w")
                    f.write( str(Text_dic) )
                    f.close()
                    g = open(OUT2,"w")
                    g.write( str(Img_dic) )
                    g.close()
                    
                    
            
        return img_positions, word_positions, Text_dic, Img_dic, SECOND_NAME, IMAGE_DETAILS, IMG_PATH
    
        
    
if __name__ == '__main__':
    BASE_FOLDER = "PRIMA_IMAGES"

    if os.path.exists(BASE_FOLDER) == False:
        os.mkdir(BASE_FOLDER)

    if len(sys.argv) != 2:
        print("Usage: NTDRW_CLASS.py <filename>")
        sys.exit(1)

    start = Document_Analysis(sys.argv[1].replace(BASE_FOLDER+"/",""), True, BASE_FOLDER)
