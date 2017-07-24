import os
import re
import pandas as pd
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.ndimage.filters import convolve
import numpy as np
try:
    import dicom
except ImportError:
    print("warning: pydicom is not installed, so you can load the png files,but not the .dcms in the large dataset ")
#Utilities for preprocessing

def remove_blank(img):
    """crops out all blank rows/columns on edges of image """
    nonblank_rows,nonblank_cols = np.where(img>0)
    rmin, rmax = nonblank_rows.min(), nonblank_rows.max()
    cmin, cmax = nonblank_cols.min(), nonblank_cols.max()
    return img[rmin:rmax,cmin:cmax]

def png_to_array(path,res=(256,256)):
    img = imread(path,flatten=True)
    if (img.max() <= 0.0):
        raise ValueError("empty image. imgname: "+ path)
    return preprocess_image(img,res)

def preprocess_image(img,res=(256,256)):
    """Set image to desired resolution and normalise to have mean 0 and std 1 """
    img = remove_blank(img)
    img = imresize(img,res)
    img = img.astype("float32") - img.mean()
    img /= img.std()
    return img


#This stuff is for loading the 2 smaller datasets
def process_reading(reading_path):
    """For the small datasets. Opens reading file specified by reading_path, parses 
    it and returns a dictionary """

    with open(reading_path,"r") as f:
        reading_lines = f.readlines()
    #find age and gender with a regex
    age_regex = "[0-9]{1,3}(?=[yY])"
    age = re.findall(age_regex, reading_lines[0] + reading_lines[1])
    gender = "M" if re.findall("[Ff]", reading_lines[0]) == [] else "F"
    print(gender, age,  reading_lines)

    #in case regex doesn't match
    if len(age) != 1 or len(gender) != 1:
        print(age)
        print(gender)
        print("problem parsing file {} with contents: {}".format(
                    reading_path, reading_lines))
        age, gender = ["-1"], ["UNK"]
    age = int(age[0])
    gender = gender[0][0].lower()

    #diagnosis is on last line for both datasets
    diagnosis = reading_lines[-1]
    info_dict = {"age": age,
                 "gender": gender,
                 "diagnosis": diagnosis,
                 "full_reading": reading_lines,
                 "path": reading_path}

    return info_dict


def basic_load(setpath,res=(256,256), img_dir="CXR_png", 
                 label_dir="ClinicalReadings"):
    print("setpath: ", setpath)
    imgfiles = sorted(os.listdir(os.path.join(setpath,img_dir)))
    labelfiles = sorted(os.listdir(os.path.join(setpath,label_dir)))
    #preallocate ndarrays for images and targets and a list for metadata
    X = np.zeros((len(imgfiles),)+res)
    y = np.zeros(len(imgfiles),)
    info = [None] * len(imgfiles)
    i = 0
    for (imname, labname) in zip(imgfiles,labelfiles):
        print("loading image: " + str(imname) + ", and labels: " + str(labname))
        impath = os.path.join(setpath,img_dir,imname)
        label_path = os.path.join(setpath,label_dir,labname)
        img = png_to_array(impath)
        info[i] = process_reading(label_path)
        X[i] = img
        if "normal" not in info[i]["diagnosis"]:
            y[i] = 1
            print(info[i]["diagnosis"])
            assert label_path[-5] == "1"
        i+=1
    return (X,y,info)

#This stuff is all for the DICOM files and labels

def change_filename(png_fn):
    return  png_fn[3:-4] + ".dcm"

def process_csv_row(entry,basedir):
    """Turns a row of csv into a (path, label) pair """
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
    X,y = [], []
    for pair in labeled_paths[4:]:
        (path,label) = pair
        filedata = dicom.read_file(path)
        try:
            #data attributes are store with 2 hex numbers for some reason
            view = filedata[(0x0018,0x5101)].value
            if view not in allowed_views:
                continue
            else:
                #filedata 
                img = filedata.pixel_array
                img=img.astype("float32")
                #Make sure it's black-on-white, not white-on-black
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
