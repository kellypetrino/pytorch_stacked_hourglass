import scipy.io
import scipy.misc
import numpy as np

# Load in annotations
datadir = 'data/MPII/'
annotpath = 'data/MPII/annot/mpii_human_pose_v1_u12_1.mat'
annot = scipy.io.loadmat(annotpath)['RELEASE']
nimages = annot['img_train'][0][0][0].shape[0]

def imgpath(idx):
    # Path to image
    filename = str(annot['annolist'][0][0][0]['image'][idx][0]['name'][0][0])
    return datadir + '/images/' + filename

def loadimg(idx):
    # Load in image
    return scipy.misc.imread(imgpath(idx))

def getAllCats():
    # Iterate thru all images and make dictionary of img names and labels
    categories = {}
    for idx in range(nimages):
        # Get the category label and imgname for the image 
        if annot['act'][0][0][idx]['act_id'][0][0] != -1:
            label = annot['act'][0][0][idx]['cat_name'][0][0]
            imgname = str(annot['annolist'][0][0][0]['image'][idx][0]['name'][0][0])
            categories[imgname] = label

    return categories

            
