from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import os

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob

import gc
import time

import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchmetrics

import time
import warnings
warnings.filterwarnings("ignore")

from IPython.display import Image
from skimage import io

import segmentation_models_pytorch as smp

from pprint import pprint

from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display

from PIL import Image

import torchvision
from torchvision import transforms

from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




import pandas as pd
app = Flask(__name__)
CORS(app, supports_credentials=True)
# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OUTPUT_FOLDER = 'output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the 'file' key is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check if the file is allowed (you can add more file types if needed)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif','tif','tiff'}
        if not allowed_file(file.filename, allowed_extensions):
            return jsonify({'error': 'Invalid file type'})

        # Save the uploaded file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        
        print(file.filename)
        image_processing(file)

        return jsonify({'message': 'File uploaded successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def image_processing(file):
    import torchvision
    from torchvision import transforms
    print(file.filename)
    data = [['file.filename', f'F:/spl3/flask/uploads/{file.filename}',
           f'F:/spl3/flask/uploads/{file.filename}', 0]]
    columns = ['patient_id', 'img_path', 'mask_path', 'mask']

    mri_df = pd.DataFrame(data=data, columns=columns)

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])

    def  adjust_data(img, mask):
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        
        return (img, mask)

    class MyDataset(Dataset):
        def __init__(self, df= mri_df, 
                    adjust_data = adjust_data, 
                    image_transform=image_transform, mask_transform=mask_transform):
            self.df = df
            self.image_transform = image_transform
            self.mask_transform = mask_transform
            self.adjust_data= adjust_data

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image_path = self.df.loc[idx, 'img_path']
            mask_path = self.df.loc[idx, 'mask_path']

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path)
    #         mask =cv2.imread(mask_path, 0)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            image, mask = self.adjust_data(image, mask)

            if self.image_transform:
                image = self.image_transform(image).float()

            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, mask
        
    print(len(mri_df))
    traids = MyDataset(df=mri_df)
    train_loader = DataLoader(traids, batch_size=1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    class Block(nn.Module):
        def __init__(self, inputs = 3, middles = 64, outs = 64):
            super().__init__()
            #self.device = device
            #self.dropout = nn.Dropout(dropout)
            
            self.conv1 = nn.Conv2d(inputs, middles, 3, 1, 1)
            self.conv2 = nn.Conv2d(middles, outs, 3, 1, 1)
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(outs)
            self.pool = nn.MaxPool2d(2, 2)
            
        def forward(self, x):
            
            x = self.relu(self.conv1(x))
            x = self.relu(self.bn(self.conv2(x)))
            # e1 = x
            # x = self.pool(x)
            
            return self.pool(x), x
            # self.pool(x): [bs, out, h*.5, w*.5]
            # x: [bs, out, h, w]    
        
            # return x, e1
            # x: [bs, out, h*.5, w*.5]
            # e1: [bs, out, h, w]
    class UNet(nn.Module):
        def __init__(self,):
            super().__init__()
            #self.device = device
            #self.dropout = nn.Dropout(dropout)
            
            self.en1 = Block(3, 64, 64)
            self.en2 = Block(64, 128, 128)
            self.en3 = Block(128, 256, 256)
            self.en4 = Block(256, 512, 512)
            self.en5 = Block(512, 1024, 512)
            
            self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride = 2)
            self.de4 = Block(1024, 512, 256)
            
            self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride = 2)
            self.de3 = Block(512, 256, 128)
            
            self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride = 2)
            self.de2 = Block(256, 128, 64)
            
            self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
            self.de1 = Block(128, 64, 64)
            
            self.conv_last = nn.Conv2d(64, 1, kernel_size=1, stride = 1, padding = 0)
            
        def forward(self, x):
            # x: [bs, 3, 256, 256]
            
            x, e1 = self.en1(x)
            # x: [bs, 64, 128, 128]
            # e1: [bs, 64, 256, 256]
            
            x, e2 = self.en2(x)
            # x: [bs, 128, 64, 64]
            # e2: [bs, 128, 128, 128]
            
            x, e3 = self.en3(x)
            # x: [bs, 256, 32, 32]
            # e3: [bs, 256, 64, 64]
            
            x, e4 = self.en4(x)
            # x: [bs, 512, 16, 16]
            # e4: [bs, 512, 32, 32]
            
            _, x = self.en5(x)
            # x: [bs, 512, 16, 16]
            
            
            x = self.upsample4(x)
            # x: [bs, 512, 32, 32]
            x = torch.cat([x, e4], dim=1)
            # x: [bs, 1024, 32, 32]
            _,  x = self.de4(x)
            # x: [bs, 256, 32, 32]
            
            x = self.upsample3(x)
            # x: [bs, 256, 64, 64]
            x = torch.cat([x, e3], dim=1)
            # x: [bs, 512, 64, 64]
            _, x = self.de3(x)
            # x: [bs, 128, 64, 64]
            
            x = self.upsample2(x)
            # x: [bs, 128, 128, 128]
            x = torch.cat([x, e2], dim=1)
            # x: [bs, 256, 128, 128]
            _, x = self.de2(x)
            # x: [bs, 64, 128, 128]
            
            x = self.upsample1(x)
            # x: [bs, 64, 256, 256]
            x = torch.cat([x, e1], dim=1)
            # x: [bs, 128, 256,256, 256
            _, x = self.de1(x)
            # x: [bs, 64, 256, 256]
            
            x = self.conv_last(x)
            # x: [bs, 1, 256, 256]
            
            # x = x.squeeze(1)         
            return x
            
    class UNETModel(pl.LightningModule):

        def __init__(self,):
            super().__init__()
            self.model = UNet()

            # for image segmentation dice loss could be the best first choice
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        def forward(self, image):
            # normalize image here
            # image = (image - self.mean) / self.std
            mask = self.model(image)
            return mask

        def shared_step(self, batch, stage):
            
            image = batch[0]

            # Shape of the image should be (batch_size, num_channels, height, width)
            # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
            assert image.ndim == 4

            # Check that image dimensions are divisible by 32, 
            # encoder and decoder connected by skip connections and usually encoder have 5 stages of 
            # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
            # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
            # and we will get an error trying to concat these features
            h, w = image.shape[2:]
            assert h % 32 == 0 and w % 32 == 0

            mask = batch[1]

            # Shape of the mask should be [batch_size, num_classes, height, width]
            # for binary segmentation num_classes = 1
            assert mask.ndim == 4

            # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
            assert mask.max() <= 1.0 and mask.min() >= 0

            logits_mask = self.forward(image)
            
            # Predicted mask contains logits, and loss_fn param from_logits is set to True
            loss = self.loss_fn(logits_mask, mask)

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then 
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
            
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])
            
            total_loss = 0
            iter_count = len(outputs)
        
            for idx in range(iter_count):
                total_loss += outputs[idx]['loss'].item()

            # per image IoU means that we first calculate IoU score for each image 
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset 
            # with "empty" images (images without target class) a large gap could be observed. 
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            
            metrics = {
                f"{stage}_loss": total_loss/iter_count,
                f"{stage}_precision": precision,
                f"{stage}_recall": recall,
                f"{stage}_accuracy": accuracy,
                f"{stage}_f1_score": f1_score,
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }
            
            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, "train")            

        def training_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, "valid")

        def validation_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "valid")

        def test_step(self, batch, batch_idx):
            return self.shared_step(batch, "test")  

        def test_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0001)
        
        
    model= UNETModel().to(device)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    check_path = r'F:\spl3\model_best.ckpt'
    # ckpt = torch.load(check_path, map_location=device)
    # print(ckpt.keys())
    # model = model.load_state_dict(ckpt['state_dict'])
    model = UNETModel.load_from_checkpoint(check_path, map_location=device) 


    # print(help(model))
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print(type(train_loader))
    x = iter(train_loader)
    batch = next(x)
    print('shape', batch[0].shape)
    # batch = next(iter(train_loader))
    # print(help(train_loader))
    # image_path =r'F:\galib\spl3\flask\uploads\TCGA_CS_4941_19960909_1.tif'

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image_transform(image).float()
    # print(len(train_loader))

    # img = cv2.imread(r'K:\Sikdar\spl3\flask\uploads\TCGA_CS_4941_19960909_1.tif')
    # img = Image.open(r'F:\spl3\flask\uploads\TCGA_CS_4941_19960909_12.tif')
    print(len(train_loader))
    with torch.no_grad():
        ans = model(image=batch[0])
    pr_masks = (ans.sigmoid() > .5).float()
    im = pr_masks[0][0].numpy()*255
    print(im.shape)
    cv2.imwrite('ans.jpg', im)
    processed_image_path = os.path.join(app.config['OUTPUT_FOLDER'],'ans.jpg')
    # Return the processed image
    return send_file(processed_image_path)
if __name__ == '__main__':
    app.run(debug=True)
