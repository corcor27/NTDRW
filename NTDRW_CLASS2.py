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
from googletrans import Translator
import sys

class Document_Analysis():
    def __init__(self, Doc_name, OLD, folder):
        self.folder = folder
        self.Doc_name = Doc_name
        self.OLD = OLD
        ### Copy image/ convert pdf, and return image list and folder paths
        self.Img_list, self.Root_folder, self.Document_folder, self.Image_files_folder, self.Image_details_folder, self.Masks_folder, self.Samples_folder = self.Generate_Copy()
        self.IMG_PATH = self.Analyse_document()
        #self.IMG_PATH = os.path.join(self.Image_files_folder, self.Img_list[0])
        self.lang = self.Check_welsh()
        if self.lang == "cy":
            None
            #self.Bing()
        self.Text_summary()
        self.word_understanding()
        #self.SEG = self.Image_segmentation()
        
        
    def Check_welsh(self):
        for item in self.Img_list:
            strip_ext = os.path.splitext(item)[0]
            Text_file_path = os.path.join(self.Image_details_folder, "text_details.txt")
            with open(Text_file_path) as f:
                data = f.read()
            data = ''.join(data)
            #print(data)
            if data != "":
                translator = Translator()
                langs = translator.detect(data)
        print(langs.lang)
        return langs.lang
            
        
        
    def Bing(self):
        for item in self.Img_list:
            strip_ext = os.path.splitext(item)[0]
            Text_file_path = os.path.join(self.Image_details_folder, "text_details.txt")
            with open(Text_file_path) as f:
                data = f.read()
            print(data)
            if data != "":
                translator = Translator()
                translation = translator.translate(data, dest='en')
                print(translation.text)
                tran_out = os.path.join(self.Image_details_folder, "text_details.txt")
                os.remove(Text_file_path)
                f = open(tran_out,"w")
                f.write( str(translation.text) )
                f.close()
        return 
                
            
        
    def Image_segmentation(self):
        image = Image.open(self.IMG_PATH).convert('RGB')
        IMG_COL = cv2.imread(self.IMG_PATH)
        image2 = ImageDraw.Draw(image)
        text_key_base = {}
        feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
        model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
           
        inputs = feature_extractor(images=IMG_COL, return_tensors="pt")
        outputs = model(**inputs)
        # model predicts COCO classes, bounding boxes, and masks
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        text_size = 15
        font = ImageFont.load_default()
        for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
            shape = [(xmin, ymin), (xmax, ymax)]
            cl = p.argmax()
            image2.rectangle(shape,  outline ="red")
            image2.text((xmin, ymin), "{}".format(model.config.id2label[cl.item()]), fill=(0, 0, 0), font=font)
            print(model.config.id2label[cl.item()])
        image.save(os.path.join(self.Image_details_folder, "base2.png"))
        return 0
    
    def Image_analysis(self, POSITIONS, IMG_PATH):
        
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
    
    
    def model_predict(self, IMG_PATH):
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5, det_algorithm="DB") 
        result = ocr.ocr(IMG_PATH, cls=True)
        return result
    
            
    def Text_summary(self):
        for item in self.Img_list:
            strip_ext = os.path.splitext(item)[0]
            Text_file_path = os.path.join(self.Image_details_folder, "text_details.txt")
            with open(Text_file_path) as f:
                data = f.read()
            #print(data)
            if data != "":
                if len(data) >= 1024:
                    data = data[0:1024]

                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                OUTPUT = summarizer(data, max_length=200, min_length=10, do_sample=False)
                OUT2 = os.path.join(self.Image_details_folder, "Summary.txt")
                print(OUT2)
                #analysis_text = self.word_understanding(Text_dic)
                # open file for writing
                print(OUTPUT)
                f = open(OUT2,"w")
                f.write(str(OUTPUT[0]['summary_text']))
                f.close()
            return 0
    
    def word_understanding(self):
        for item in self.Img_list:
            strip_ext = os.path.splitext(item)[0]
            Text_file_path = os.path.join(self.Image_details_folder, "text_details.txt")
            with open(Text_file_path) as f:
                data = f.read()
            Text = data
            tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                # make example sentence
            sentence = Sentence(Text)
                # predict NER tags
            tagger.predict(sentence)
            words = []
            for entity in sentence.get_spans('ner'):
                en = str(entity)
                en = en.split(" ")
                words.append(en)
            Text_details_path = os.path.join(self.Image_details_folder, "Importance.txt")
            f = open(Text_details_path,"w")
            f.write(str(words))
            f.close()
                
                
                
        
    def Text_analysis(self, POSITIONS, IMG_PATH):
        words_key_base = {}
        img = cv2.imread(IMG_PATH,0)
        img2 = cv2.imread(IMG_PATH)
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        image64 = Image.open(IMG_PATH).convert('RGB')
        image68 = ImageDraw.Draw(image64)
        
        for jj in range(0, POSITIONS.shape[0]):
            yy, xx, h, w = POSITIONS[jj,:]
            Samples_out = os.path.join(self.Samples_folder, "{}_{}_{}_{}.png".format(yy, xx, h, w))
            cv2.imwrite(Samples_out, img2[yy:yy+h, xx:xx+w, :])
            result = self.model_predict(img2[yy:yy+h, xx:xx+w, :])
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
                pixel_values = processor(images=base, return_tensors="pt").pixel_values
                    
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                words_key.append(generated_text)
                image68.rectangle(shape,  outline ="red")
                image68.text((x0+xx, y0+yy-text_size), "{}".format(generated_text), fill=(0, 0, 255), font=font)
            words_key_base["{}_{}_{}_{}".format(yy, xx, h, w)] = words_key
           
        #image64.save(OUT1)
        return words_key_base
        
    def Word_Extraction(self, IMG_PATH):
        gray_img = cv2.imread(IMG_PATH, 0)
        color_img = cv2.imread(IMG_PATH)
        ret, gray_img2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        IMG_PATH2 = os.path.join(self.Image_files_folder, "gray.png")
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
            convert_from_path(PDF_PATH, dpi=200, output_folder=OUT, first_page=page, last_page = min(page+10-1,maxPages), fmt="PNG" )
        IMG_LIST = os.listdir(OUT)
        return IMG_LIST
        
    def Generate_Copy(self):
        Doc_convert = "Doc_analysis"
        if os.path.exists(Doc_convert) == False:
            os.mkdir(Doc_convert)
        File_folder = os.path.join(Doc_convert, self.Doc_name)
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
        Samples_folder = os.path.join(File_folder, "Text_samples")
        if os.path.exists(Samples_folder) == False:
            os.mkdir(Samples_folder)
        if "pdf" in self.Doc_name:
            if self.OLD == True:
                IMG_PATH_LIST = self.convert_pdf(self.Doc_name, IMAGE_SAVE)
        
        else:
            if ".tif" in self.Doc_name:
                IMG_PATH = os.path.join(IMAGE_SAVE, self.Doc_name.replace("tif", "png"))
                shutil.copy("{}/{}".format(self.folder, self.Doc_name), IMG_PATH)
                IMG_PATH_LIST = [self.Doc_name.replace("tif", "png")]
            else:
                IMG_PATH = os.path.join(IMAGE_SAVE, self.Doc_name)
                shutil.copy("{}/{}".format(self.folder, self.Doc_name), IMG_PATH)
                IMG_PATH_LIST = [self.Doc_name]
        return IMG_PATH_LIST, Doc_convert, File_folder, IMAGE_SAVE, IMAGE_DETAILS, IMAGE_OUT, Samples_folder
    
    def Analyse_document(self):
        print(self.Img_list)
        for item in self.Img_list:
            strip_ext = os.path.splitext(item)[0]
            IMG_PATH = os.path.join(self.Image_files_folder, item)
            IMG_MASK_PATH = os.path.join(self.Masks_folder, "Img_mask" + item)
            TEXT_MASK_PATH = os.path.join(self.Masks_folder, "Text_mask" + item)
            img_positions, img_mask = self.Image_Extraction(IMG_PATH)
            word_positions, word_mask = self.Word_Extraction(IMG_PATH)
            Text_dic = self.Text_analysis(word_positions, IMG_PATH)
            cv2.imwrite(IMG_MASK_PATH, img_mask)
            cv2.imwrite(TEXT_MASK_PATH, word_mask)
            Img_dic = self.Image_analysis(img_positions,IMG_PATH)
            Text_details_path = os.path.join(self.Image_details_folder, "text_detail_dic.txt")
            Img_details_path = os.path.join(self.Image_details_folder, "img_detail_dic.txt")
            Text_details_path2 = os.path.join(self.Image_details_folder, "text_details.txt")
            
            f = open(Text_details_path,"w")
            f.write(str(Text_dic))
            f.close()
            g = open(Img_details_path,"w")
            g.write(str(Img_dic))
            g.close()
            Text = []
            #print(Text_dic)
            key = Text_dic.keys()
            for opt in key:
                Section = Text_dic[opt]
                for txt in Section:
                    Text.append(txt)
                    
            Text = ''.join(Text)
            Text = Text.strip('"')
            h = open(Text_details_path2,"w")
            h.write(str(Text))
            h.close()
            
                    
        return IMG_PATH

if __name__ == '__main__':
    BASE_FOLDER = "IMG"

    if os.path.exists(BASE_FOLDER) == False:
        os.mkdir(BASE_FOLDER)

    if len(sys.argv) != 2:
        print("Usage: NTDRW_CLASS.py <filename>")
        sys.exit(1)

    start = Document_Analysis(sys.argv[1].replace(BASE_FOLDER+"/",""), True, BASE_FOLDER)
