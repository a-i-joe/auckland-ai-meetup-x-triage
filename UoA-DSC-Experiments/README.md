# University of Auckland Data Science Club X-Traige Entry

To get the ball rolling, here's the code for our experiments so far. It's more a proof of concept to build on than a finished product.
Includes scripts to preprocess the data and train a (very) basic convolutional neural network to classify x-rays into "normal" or "abnormal", and utils for running cross validation and finding auc.

## Details
So far, it only trains on the two smaller datasets. I did load the large DICOM dataset and the labels from the scraper, but was unable to get the same model to learn anything - performed no differently to random labels. I think this is because either:
- a) Something is wrong with my code to load data and labels - the functions for loading are still in prepro.py
- b) Something went wrong with Robin's scraper
- c) Something is going wrong with the REST API - I did notice the filenames from the API were for .pngs, whereas the data itself was in DICOM - odd.

The data has a lot of other issues as well, such as multiple x-ray views(side-on vs front-on) that have incorrect labels in the DICOM file header.

The model was trained on a gtx970 with a pentium g4400, and took <1 second per epoch. It's probably small enough to train on CPU.

## Results
Running 10x10-fold cross validation it averages an auc of 0.85 ± 0.07. Here is an example of what the roc curves look like:
![roc](images/figure_1.png)
Estimating the safe set % is difficult and unreliable when there is so little data. The model filtered out on average 27% of normal x-rays while getting a single false negative(out of 80 val examples). This was tested with 10 way cross validation, and there was a lot of variance in safe set% across different splits - 10% to 65%.

This probably isn't good enough to be useful yet, but it's a pretty decent first step considering this is only using a tenth of the avaliable data and the model trains in ~8 seconds.

## Steps to reproduce
Everything was tested on Ubuntu 16.04 with python 2.7.12, gcc 5.4.0, with cuda 8 and cudnn 5, but should work on any unix system.
### Installing dependencies
This uses the standard scientific python stack(scipy+numpy+matplotlib),scikit-learn,seaborn and the deep learning library [keras](keras.io).
 All dependencies can be installed with pip(you might have to run pip as root):
```bash
git clone https://github.com/UOADataScience/auckland-ai-meetup-x-triage
cd auckland-ai-meetup-x-triage/UoA-DSC-Experiments
pip install -r requirements.txt
```
### GPU
If you want to train on GPU, make sure you also have the NVIDIA drivers, a recent version of CUDA and CUDNN installed and tensorflow or theano is built with GPU support.
### Preprocessing data, building a model and training:
First, download and unzip the 2 smaller datasets:
- https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

Run the script to preprocesses the data, storing the preproccesed data in SAVEPATH. --res allows you to optionally set the resolution of data and defaults to (256,256)
```bash
python prepare_smalldata.py --dir /path/to/ChinaSet_AllFiles /path/to/MontgomerySet --savepath SAVEPATH --res 256 256
```
Now the data is pickled in a tuple of a ndarray of images and a list of labels, ready for a neural net. To load it into python, simply use:
```python
import dill as pickle
with open(SAVEPATH,"r") as f:
	(X,y) = pickle.load(f)
```
To run 10-way cross validation and plot the ROC:
```bash
python train.py --datapath SAVEPATH
```

## TODO
- Figure out what I've done wrong with the large dicom datasets labels (priority: very high)
- plot safe set % vs dataset size
- Test on OSX
- Test on Windows - might require a few modifications to work
- Write script to get this all to work on an AWS EC2 instance
- Modify code for loading labels so that it also loads a list of ages and genders for each image, explore their relationship with model performance
- Augment the dataset with random skews, shifts, crops and rotations(look up keras' ImageDataGenerator class)
- Add ability to monitor AUC during training (probably using a keras callback)
- Write keras callback to monitor "safe set" %
- Write keras callback to save weights that perform best in one of the above metrics
- Try out some different neural net architectures.
- Write code to plot [saliency maps](https://arxiv.org/pdf/1312.6034.pdf) or  other visualization methods to help interpret how model makes predictions
- Refactor code so it's cleaner/faster/pylint compliant 
