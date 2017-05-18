import os
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import dicom
except ImportError:
    print("warning: pydicom is not installed, so you can load the png files,but not the .dcms in the large dataset ")
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np

#Utilities for preprocessing

def remove_blank(img):
    """crops out all blank rows/columns on edges of image """
    nonblank_rows,nonblank_cols = np.where(img>0)
    rmin, rmax = nonblank_rows.min(), nonblank_rows.max()
    cmin, cmax = nonblank_cols.min(), nonblank_cols.max()
    return img[rmin:rmax,cmin:cmax]

dirs = ['/home/soren/Desktop/smallsets/ChinaSet_AllFiles', '/home/soren/Desktop/smallsets/MontgomerySet']#replace with argparse soon
d = '/home/soren/Desktop/smallsets/'


def png_to_array(path,res=(256,256)):
    img = imread(path,flatten=True)
    return preprocess_image(img,res)

def preprocess_image(img,res=(256,256)):
    """Set image to desired resolution and normalise to have mean 0 and std 1 """
    img = remove_blank(img)
    img = imresize(img,res)
    img = img.astype("float32") - img.mean()
    img /= img.std()
    return img
#This stuff is for loading the 2 smaller datasets

def load_labels(path):
    gender =0
    abnormal  = 1
    text = open(path,"r").readlines()
    if "female" in text:
        gender = 1
    if "normal" in text:
        abnormal = 0
    age = int("".join([c for c in text[0] if c.isdigit()]))
    print("age:{}".format(age))
    return (age,gender,abnormal)


def basic_load(setpath,res=(256,256), img_dir="CXR_png",label_dir="ClinicalReadings"):
    imgfiles = sorted(os.listdir(os.path.join(setpath,img_dir)))
    labelfiles = sorted(os.listdir(os.path.join(setpath,label_dir)))
    X = np.zeros((len(imgfiles),)+res)
    y = np.zeros(len(imgfiles),)
    i=0
    for (imname, labname) in zip(imgfiles,labelfiles):
        print("loading image: "+str(imname) + ", and labels: " +str(labname))
        impath = os.path.join(setpath,img_dir,imname)
        labelpath = os.path.join(setpath,label_dir,imname)
        img = png_to_array(impath)
        X[i] = img
        if imname[-5] == "1":
            y[i] = 1
        i+=1
    return (X,y)

#This stuff is all for the DICOM files

def change_filename(png_fn):
    return  png_fn[3:-4] + ".dcm"


def process_csv_row(entry,basedir):
    label_dict = {" positive\n":1," negative\n":0}
    parts = entry.split(",")
    label = label_dict[parts[-1]]
    path_components = parts[1].split("\/")
    subdir = path_components[4]
    filename = change_filename(path_components[-1])
    path = os.path.join(basedir,subdir,filename)
    return (path,label)

def match_dicoms(basedir,labels_path):
    """The paths in the label csv don't quite match up with the data folder.this
        fn returns (path,label) pairs for loading dicoms."""
    label_csv = open(labels_path,"r").readlines()[1:-1]
    labeled_paths = []
    for entry in label_csv:
        (path,label) = process_csv_row(entry,base_dir)
        labeled_paths.append((path,label))
    return labeled_paths

def load_dicoms(base_dir,labels_path,allowed_views=["LL"],res=(256,256)):
    """Returns np arrays (X,y) ready for deep learning """
    labeled_paths = match_dicoms(base_dir,labels_path)
    X = []
    y = []
    for pair in labeled_paths[4:]:
        (path,label) = pair
        filedata = dicom.read_file(path)
        try:
            view = filedata[(0x0018,0x5101)].value
            if view not in allowed_views:
                continue
            else:
                img = filedata.pixel_array
                img=img.astype("float32")
                if filedata[(0x0028,0x0004)].value != "MONOCHROME1":
                    print("inverting")
                    img = img.max() - img
                img = preprocess_image(img,res)
                X.append(img)
                y.append(label)
        except KeyError:
            print("key error on file:")
            print(path)
    X = np.stack(X)
    X = np.expand_dims(X,axis=-1)
    y = np.asarray(y)
    return (X,y)
