import os, io
import cv2
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F

import plotly
import plotly.express as px

from torchvision import transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template, redirect, url_for
from torchvision.transforms.transforms import ToPILImage



# config params --------------------------------------
MODEL_WEIGHTS = 'WingWatchModelStateDict.pth'
CATALOG_PATH = './static/catalog'

#buffers for data persistence across requests
confidence_buf = io.BytesIO()
id_buff = io.BytesIO()

device = torch.device("cpu")

# species classification from csv
class_data = pd.read_csv('class_dict.csv')
class_data = class_data['class']

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# -----------------------------------------------------

# Image Transformations
image_transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])


app = Flask(__name__, static_url_path='/static')

# Url for Home page from 'wing-watcher-app.html'
@app.route('/')
def render_home():
    return render_template('wing-watcher-app.html')

#Url for Species Catalog from 'catalog.html'
@app.route('/catalog')
def render_catalog():
    images = os.listdir(CATALOG_PATH)
    return render_template('catalog.html', images=images)

#Url for Confidence Score Graph Tab
@app.route('/confidence_tab')
def confidence_tab():
    return render_template('confidence_tab.html')

#Construct Confidence Graph and Save as PNG for Display
@app.route('/confidence_graph')
def confidence_graph():

    if(confidence_buf.getbuffer().nbytes != 0):
        score_file = np.load(confidence_buf, allow_pickle=True, fix_imports=False)
        id_file = np.load(id_buff, allow_pickle=True, fix_imports=False)

        confidence_scores = score_file['arr_0']
        bird_ids = id_file['arr_0']
        bird_names = []
        for i in range(len(bird_ids)):
            bird_names.append(class_data[bird_ids[i]])
    
        print(bird_names)

        df = pd.DataFrame({
            "Bird Species": bird_names,
            "Classification Confidence Scores": confidence_scores
        })

        fig = px.bar(df, x="Bird Species", y="Classification Confidence Scores", color="Classification Confidence Scores")

        fig.write_image("./static/images/fig_storage.png")

        confidence_buf.seek(0)
        id_buff.seek(0)
        confidence_buf.truncate()
        id_buff.truncate()

    return

# Url for predictions using POST
@app.route('/predict',methods=['POST'])
def predict(in_img, trained_model):

    # Convert to tensor by applying transformation
    img_tensor = image_transformation(in_img)
    img_tensor = img_tensor.unsqueeze(0)
    # Load tensor in dataloader
    img_dataloader = DataLoader(img_tensor, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        input = iter(img_dataloader).next()
        input = input.to(device)

        output = trained_model(input)
        _, prediction = torch.max(output, 1)
        probs = torch.softmax(output, dim=1)
        

        top5_probs,top_ids = torch.topk(probs, 5)
        top_prob,top_class_id = torch.topk(probs,1)

        #save probability and species outputs for access across requests
        probs_save = top5_probs.squeeze(0).numpy()
        ids_save = top_ids.squeeze(0).numpy()
        
        np.savez_compressed(confidence_buf,probs_save)
        np.savez_compressed(id_buff,ids_save)

        confidence_buf.seek(0)
        id_buff.seek(0)

        confidence_graph()

        return (class_data[prediction.item()], top_prob.item())

# Ajax POST for uploading an image file for classification
@app.route('/upload',methods=['POST'])
def upload():
    
    # retrieve uploaded image and check for appropriate file format for processing
    # will accept: bmp, jpg, jpeg, rgb, png 
    in_file = request.files['file']
    img_exts = ['bmp','jpeg','jpg','png','rgb']
    
    # Split the filename on the '.' delimiter and check whether file extension is supported
    if in_file.filename.split('.')[1] not in img_exts:
        return jsonify('Please upload a supported image file: [bmp, jpg, jpeg, rgb, or png]')

    # load the pretrained application model from state dictionary
    trained_model = models.squeezenet1_1(pretrained=True)

    # turn off gradients
    for param in trained_model.parameters():
        param.requires_grad=False

    # change classifier layer[1] to fit our classification domain
    trained_model.classifier[1] = nn.Conv2d(512, 300, kernel_size=(1,1), stride=(1,1))

    # load weights and biases from state dictionary
    trained_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))    

    # Read in bytestream from imput image and convert to numpy array(uint8)
    img_byts = in_file.read()
    # pil_img = Image.open(io.BytesIO(img_byts))

    # Convert to image, format uint8
    nump_array = cv2.imdecode(np.frombuffer(img_byts,np.int8), cv2.IMREAD_COLOR)

    trained_model.to(device)
    
    prediction = predict(nump_array,trained_model)
    
    return jsonify('This is a picture of a {} with a confidence level of {:.5f}'.format(*prediction))

if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))
