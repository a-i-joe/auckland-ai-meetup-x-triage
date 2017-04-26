# UoA Data Science Club X-Traige Entry

To get the ball rolling, here's the code for our experiments so far. It's more a proof of concept to build on than a finished product.
Includes scripts to preprocess the data and train a (very) basic convolutional neural network to classify 256x256 x-rays into "normal" or "abnormal", and utils for running cross validation and finding auc. It averages ~0.85 ROC auc in 10 fold cross validation. 

## Details
So far, it only trains on the two smaller datasets. I did load the large DICOM dataset and the labels from the scraper, but was unable to get the same model to learn anything - performed no differently to random labels. I think this is because either:
a) Something went wrong with my code to load data and labels - the functions for loading are still in prepro.py
b) Something went wrong with Robin's scraper
c) Something is going wrong with the REST API - I did notice the API filenames were for .pngs, whereas the data itself was in DICOM - odd.

The data has a lot of other issues as well, such as multiple x-ray views(side-on vs front-on) that have incorrect labels in the DICOM file.

The model was trained on a gtx970 with a pentium g4400, and took <1 second per epoch. It's probably small enough to train on CPU.

## Results
Running 10-fold cross validation gets an average auc of ~0.85. Here is an example of what the roc curves look like:
![roc](images/figure_1.png)
Finding the "safe set" percentage is difficult when the dataset is small, but 

## Steps to reproduce
Everything was tested on Ubuntu 16.04 with python 2.7.12, gcc 5.4.0, with cuda 8 and cudnn 5
## Installing dependencies
This uses the standard scientific python stack(scipy+numpy+matplotlib),scikit-learn,seaborn and the deep learning library [keras](keras.io).
 All dependencies can be installed with pip:
```bash
git clone https://github.com/UOADataScience/auckland-ai-meetup-x-triage
cd auckland-ai-meetup-x-triage/UoA-DSC-entry
pip install -r requirements.txt
```
If you want to train on GPU, make sure you also have the NVIDIA drivers, a recent version of CUDA and CUDNN installed and tensorflow is built with GPU support.
## Preprocessing data, building a model and training:
First, download the 2 smaller datasets: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip,  https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
```bash
python prepro.py --dir DIR1 DIR2 --savepath SAVEPATH --res 256 256
python train.py --datapath SAVEPATH
```
To run 10-way cross validation:
```bash
python crossval.py --datapath SAVEPATH
```

## TODO
- Test on OSX
- Test on Windows - might require a few modifications to work
- Refactor code so it's cleaner/faster/pylint compliant so we look proffessional
- Augment the dataset with random skews, shifts, crops and rotations(look up keras.ImageDataGenerator)
- Add ability to monitor AUC during training (probably using a keras callback)
- Write keras callback to monitor 
- Figure out what's going on with the large datasets labels (priority: very high)
