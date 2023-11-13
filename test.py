import numpy as np
import tiffile as tiff
import datetime
import matplotlib.pyplot as plt
import cv2
import swinconvnet
import pix2pixgan
import transunet
import configparser
import torch

def Dicescore(pre, label): # Pre has a range between 0 and 1, label is binary image
    combined_region = pre * label
    precision = (np.sum(combined_region)+1e-8) / (np.sum(pre)+1e-8)
    recall = (np.sum(combined_region)+1e-8) / (np.sum(label)+1e-8)
    dice = (2 * np.sum(combined_region) +1e-8) / (np.sum(pre) + np.sum(label) + 1e-8)
    return dice, precision, recall


def image_split(input_image, batch_size):
    '''
    This function gives the start positions in three dimensions to select out patches to give prediction
    '''
    z, y, x = input_image.shape
    if z%batch_size !=0:
        zlist = batch_size * np.arange(int(z/batch_size) + 1)
    else:
        zlist = batch_size * np.arange(int(z/batch_size))

    ylist = 512 * np.arange(int(y/512))
    if y%512 !=0:
        ylist = np.append(ylist, y-512)
    xlist = 512 * np.arange(int(x/512))
    if x%512 !=0:
        xlist = np.append(xlist, x-512)
    return zlist, ylist, xlist


def model_pre(model, device, input_image, batch_size):
    z, y, x = input_image.shape
    output_image = np.zeros((z, 1, y, x))
    zlist, ylist, xlist = image_split(input_image, batch_size)
    input_image = input_image.reshape((z, 1, y, x))
    for i in range(len(zlist)):
        startz = zlist[i]
        endz = startz + batch_size
        if i == len(zlist)-1:
            endz = z
        for j in range(len(ylist)):
            starty = ylist[j]
            endy = ylist[j] + 512
            for k in range(len(xlist)):
                startx = xlist[k]
                endx = xlist[k] + 512
                middle_input = torch.tensor(input_image[startz:endz, :, starty:endy, startx:endx]).to(device)
                output_image[startz:endz, :, starty:endy, startx:endx] = model(middle_input).cpu().detach().numpy()
    output_image[output_image>0.9] = 1
    return output_image


config = configparser.ConfigParser()
config.read('config.txt')
modelname = config['model']['modelname']
modelpath = config['model']['model_path']
if_label = config.getboolean('data', 'if_label')
input_path = config['data']['input_path']
if if_label==True:
    label_path = config['data']['label_path']
pre_path = config['data']['pre_path']
batch_size = config.getint('data', 'batch_size')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(modelpath+modelname+'.pt').to(device)
image = tiff.imread(input_path)

imageshape = image.shape 
if len(imageshape) == 2:
    image = image.reshape((1, imageshape[0], imageshape[1]))
output_image = model_pre(model, device, image, batch_size)
output_image = output_image.reshape((imageshape))
tiff.imwrite(pre_path+modelname+'_pre.tif', output_image)

if if_label==True:
    label = tiff.imread(label_path)
    label[label>0.2] = 1
    label[label<1] = 0
    dice, precision, recall = Dicescore(output_image, label)
    print('The dice score is ', dice)
