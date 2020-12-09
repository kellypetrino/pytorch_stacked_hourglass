import cv2
import torch
import tqdm
import os
import numpy as np
import h5py
import copy
from imageio import imread
import sys

from OurNet import HeatNet1
from OurNet import HeatNet2
from OurNet import ImNet1
from OurNet import ImNet2
from OurNet import BothNet1
from OurNet import BothNet2

from Trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.group import HeatmapParser
import utils.img
from labels import getAllCats

# Paths
annot_dir = 'data/MPII/annot'
img_dir = 'data/MPII/images'

# Config variables
input_res = 256
output_res = 64
num_parts = 16

# Part reference
parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}

# Categories reference
cats = {'bicycling': 0, 'conditioning exercise': 1, 'dancing': 2, 'fishing and hunting': 3, 
        'home activities': 4, 'home repair': 5, 'inactivity quiet/light': 6, 
        'lawn and garden': 7, 'miscellaneous': 8, 'music playing': 9, 'occupation': 10,
        'religious activities': 11, 'running': 12, 'self care': 13, 'sports': 14, 
        'transportation': 15, 'volunteer activities': 16, 'walking': 17, 
        'water activities': 18, 'winter activities': 19}


# Preprocess images
def preprocess(data):
    # random hue and saturation
    data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    delta = (np.random.random() * 2 - 1) * 0.2
    data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

    delta_sature = np.random.random() + 0.5
    data[:, :, 1] *= delta_sature
    data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
    data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

    # adjust brightness
    delta = (np.random.random() * 2 - 1) * 0.3
    data += delta

    # adjust contrast
    mean = data.mean(axis=2, keepdims=True)
    data = (data - mean) * (np.random.random() + 0.5) + mean
    data = np.minimum(np.maximum(data, 0), 1)
    return data

# Generating a heatmap for an image at index idx
def loadImage(center, scale, part, visible, normalize, imgname):        
    ## load + crop
    path = os.path.join(img_dir, imgname)
    orig_img = imread(path)
    
    kp2 = np.insert(part, 2, visible, axis=1)
    kps = np.zeros((1, 16, 3))
    kps[0] = kp2
    orig_keypoints = kps
    
    kptmp = orig_keypoints.copy()

    c = center
    s = scale
    normalize = normalize
    
    cropped = utils.img.crop(orig_img, c, s, (input_res, input_res))
    for i in range(np.shape(orig_keypoints)[1]):
        if orig_keypoints[0,i,0] > 0:
            orig_keypoints[0,i,:2] = utils.img.transform(orig_keypoints[0,i,:2], c, s, (input_res, input_res))
    keypoints = np.copy(orig_keypoints)
    
    ## augmentation -- to be done to cropped image
    height, width = cropped.shape[0:2]
    center = np.array((width/2, height/2))
    scale = max(height, width)/200

    aug_rot=0
    
    aug_rot = (np.random.random() * 2 - 1) * 30.
    aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
    scale *= aug_scale
        
    mat_mask = utils.img.get_transform(center, scale, (output_res, output_res), aug_rot)[:2]

    mat = utils.img.get_transform(center, scale, (input_res, input_res), aug_rot)[:2]
    inp = cv2.warpAffine(cropped, mat, (input_res, input_res)).astype(np.float32)/255
    keypoints[:,:,0:2] = utils.img.kpt_affine(keypoints[:,:,0:2], mat_mask)
    if np.random.randint(2) == 0:
        inp = preprocess(inp)
        inp = inp[:, ::-1]
        keypoints = keypoints[:, flipped_parts['mpii']]
        keypoints[:, :, 0] = output_res - keypoints[:, :, 0]
        orig_keypoints = orig_keypoints[:, flipped_parts['mpii']]
        orig_keypoints[:, :, 0] = input_res - orig_keypoints[:, :, 0]
    
    ## set keypoints to 0 when were not visible initially (so heatmap all 0s)
    for i in range(np.shape(orig_keypoints)[1]):
        if kptmp[0,i,0] == 0 and kptmp[0,i,1] == 0:
            keypoints[0,i,0] = 0
            keypoints[0,i,1] = 0
            orig_keypoints[0,i,0] = 0
            orig_keypoints[0,i,1] = 0
    
    ## generate heatmaps on outres
    heatmaps = genHeatMaps(keypoints) 
    # heatmaps =  None 
    
    return inp.astype(np.float32), heatmaps

# Generate heatmaps
def genHeatMaps(keypoints):
    # Init
    sigma = output_res/64
    size = 6*sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3*sigma + 1, 3*sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Call
    hms = np.zeros(shape = (output_res, output_res, num_parts), dtype = np.float32)
    for p in keypoints:
        for idx, pt in enumerate(p):
            if pt[0] > 0: 
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=output_res or y>=output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], output_res)
                aa,bb = max(0, ul[1]), min(br[1], output_res)
                hms[aa:bb,cc:dd, idx] = np.maximum(hms[aa:bb,cc:dd, idx], g[a:b,c:d])
    return hms

# Generate and save heatmaps and processed images to machine
def saveData():
    # Load in the category labels 
    labels = getAllCats()

    # Load h5 files
    train_f = h5py.File(os.path.join(annot_dir, 'train.h5'), 'r')
    val_f = h5py.File(os.path.join(annot_dir, 'valid.h5'), 'r')
    
    # Iterate through images and generate heatmaps
    train_hms = np.zeros((len(train_f['center']), output_res, output_res, num_parts))
    train_img = np.zeros((len(train_f['center']), input_res, input_res, 3))
    bool_train = [False]*len(train_f['center'])
    train_cats = []
    for i in range(len(train_f['center'])):
        print(i)
        center = train_f['center'][i]
        scale = train_f['scale'][i]
        part = train_f['part'][i]
        visible = train_f['visible'][i]
        normalize = train_f['normalize'][i]
        imgname = train_f['imgname'][i].decode('UTF-8')
        if imgname in labels:
            train_img[i, :, :, :], _ = loadImage(center, scale, part, visible, normalize, imgname)
            bool_train[i] = True
            train_cats.append(cats[labels[imgname]])
    
    train_hms = train_hms[bool_train, :, :, :]
    train_hms = torch.tensor(train_hms, requires_grad=True)
    train_hms = train_hms.permute(0, 3, 1, 2)
    torch.save(train_hms, 'hms_train.pt')

    # Commented out code had to process heat maps and ims seperately for cpu memory issues
    # np.save('img_train.npy', train_img)
    # train_img = np.load('img_train.npy')
    train_img = train_img[bool_train, :, :, :]
    print('resizing...')
    new_img_train = np.zeros((len(train_cats), 64, 64, 3))
    for i in range(len(train_cats)):
        new_img_train[i, :, :, :] = cv2.resize(train_img[i, :, :, :], (64, 64))
    np.save('resized_img_train', new_img_train)
    train_img = new_img_train
    print('torching...')
    train_img = torch.tensor(train_img, requires_grad=True).float()
    train_img = train_img.permute(0, 3, 1, 2)
    torch.save(train_img, 'resized_img_val.pt')

    train_cats = np.array(train_cats)
    train_cats = torch.tensor(train_cats, requires_grad=False)
    torch.save(train_cats, 'train_cats.pt')
    print('train cats saved')

    # Read in validation data
    val_hms = np.zeros((len(val_f['center']), output_res, output_res, num_parts))
    val_img = np.zeros((len(val_f['center']), input_res, input_res, 3))
    bool_val = [False]*len(val_f['center'])
    val_cats = []
    for i in range(len(val_f['center'])):
        print(i)
        center = val_f['center'][i]
        scale = val_f['scale'][i]
        part = val_f['part'][i]
        visible = val_f['visible'][i]
        normalize = val_f['normalize'][i]
        imgname = val_f['imgname'][i].decode('UTF-8')
        if imgname in labels:
            val_img[i, :, :, :], val_hms[i, :, :, :] = loadImage(center, scale, part, visible, normalize, imgname)
            bool_val[i] = True
            val_cats.append(cats[labels[imgname]])
    val_hms = val_hms[bool_val, :, :, :]
    val_hms = torch.tensor(val_hms, requires_grad=True)
    val_hms = val_hms.permute(0, 3, 1, 2)

    val_img = val_img[bool_val, :, :, :] 
    print('resizing...')
    new_img_val = np.zeros((len(val_cats),64, 64, 3))
    for i in range(len(val_cats)):
        new_img_val[i, :, :, :] = cv2.resize(val_img[i, :, :, :], (64, 64))
    np.save('resized_img_val', new_img_val)
    val_img = new_img_val
    val_img = torch.tensor(val_img, requires_grad=True).float()
    val_img = val_img.permute(0, 3, 1, 2)

    # Convert labels to tensors
    val_cats = np.array(val_cats)
    val_cats = torch.tensor(val_cats, requires_grad=False)
    torch.save(val_cats, 'val_cats.pt')


def main(): 
    # LOADING IN SAVED TENSORS
    print('loading hms...')
    train_hms = torch.load('hms_train.pt').float()
    val_hms = torch.load('hms_val.pt').float()
    print(train_hms.shape)
    print(val_hms.shape)
    
    print('loading cats...')
    train_cats = torch.load('train_cats.pt')
    val_cats = torch.load('val_cats.pt')
    print(train_cats.shape)
    print(val_cats.shape)

    print('loading img...')
    train_img = torch.load('resized_img_train.pt').float()
    val_img = torch.load('resized_img_val.pt').float()
    print(train_img.shape)
    print(val_img.shape)

    # Fixing 20s to  19s in categories (quick fix to a previous typo bug)
    train_cats[train_cats == 20] = 19
    val_cats[val_cats == 20] = 19

    # code for main function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # assert('cuda' in str(device)) 

    ## Hyperparameters
    num_classes = 20  # there are 10 digits: 0 to 9
    batch_size = 128
    weight_decay = 0.0001
    lr = .001
    num_epochs = 5
    network = sys.argv[1]
    filename = f'{network}_{lr}_{weight_decay}_{batch_size}_{num_epochs}'

    ds_train = None
    ds_val = None
    both = False
    if (network == 'bn1' or network == 'bn2'):
        both = True
    
    if both:
        ds_train = TensorDataset(train_hms, train_img, train_cats)
        ds_val = TensorDataset(val_hms, val_img, val_cats)
    elif network == 'in1' or network == 'in2':
        ds_train = TensorDataset(train_img, train_cats)
        ds_val = TensorDataset(val_img, val_cats)
    else:
        ds_train = TensorDataset(train_hms, train_cats)
        print('val hms', val_hms.shape)
        print('val cats', val_cats.shape)
        ds_val = TensorDataset(val_hms, val_cats)

    # train_loader returns batches of training data
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,num_workers=0)
    val_loader = DataLoader(dataset=ds_val, batch_size=batch_size, shuffle=False,num_workers=0)

    net = None

    #Create appropriate network based on input
    if network == 'hn1':
        net = HeatNet1(num_classes)
    elif network == 'hn2':
        net = HeatNet2(num_classes) 
    elif network == 'in1':
        net = ImNet1(num_classes)
    elif network == 'in2':
        net = ImNet2(num_classes)
    elif network == 'bn1':
        net = BothNet1(num_classes)
    elif network == 'bn2':
        net = BothNet2(num_classes)

    net = net.to(device)  # move the net to the GPU
    opt = optim.Adam(net.parameters(), lr=lr)   # uses Adam optimimzer
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader, device=device, filename=filename)
    losses = None
    acc = None
    if both:
        losses, acc = trainer.train2(num_epochs)
    else:
        losses, acc = trainer.train1(num_epochs)
    np.save(f'train_loss/{filename}.train_losses.npy', np.array(losses))
    np.save(f'train_acc/{filename}.train_acc.npy', np.array(acc))



    tot = 0
    cor = 0
    if not both:
        with torch.no_grad():
            for data in val_loader:
                # Load data to gpu
                X = data[0].to(device)
                y = data[1].to(device)

                # raw output of network for X
                output = net(X)
                
                # let the maximum index be our predicted class
                _, yh = torch.max(output, 1) 

                # Running total of number of datapoints
                tot += y.size(0)

                ## Get number of correct labels
                cor += len(y[y==yh])
    else:
        with torch.no_grad():
            for data in val_loader:
                # Load data to gpu
                X = data[0].to(device)
                Y = data[1].to(device)
                z = data[1].to(device)

                # raw output of network for X
                output = net(X, Y)
                
                # let the maximum index be our predicted class
                _, zh = torch.max(output, 1) 

                # Running total of number of datapoints
                tot += z.size(0)

                ## Get number of correct labels
                cor += len(z[z==zh])        

    acc = cor/tot * 100
    print(f'*** Val acc: {acc} ***')
    np.save(f'val_acc/{filename}_{acc:.3f}.npy', np.array([acc]))   # Save validation accuracy
    torch.save(net.state_dict(), f'our_models/{filename}.{acc:.3f}.pt') # Save network labeled with val accuracy


        
if __name__ == '__main__':
    main()