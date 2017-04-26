import numpy as np
from util import *
from keras.layers import Input,Conv2D,MaxPool2D,Dense,Flatten,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD,Adadelta
from keras.models import Sequential
from keras.preprocessing import image
import dill as pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

#f = h5py.File("uncleaned_data.h5",mode="r")
#X = f["X_unclean"]
#X = X[:,:,:]
#X = X.reshape(X.shape[0],256,256,1)
#f1 = h5py.File("labels_data__uncleaned.h5")
#y = f1["uncleaned_data_labels"]
#y = y[:]
#y = [float(o) for o in y]
#y = np.asarray(y)
save_path ="/home/soren/Desktop/test.pickle" 
#save_path ='/home/soren/Desktop/smallsets/lateral_xy.pickle'  
with open(save_path,"r") as f:
    (X,y) = pickle.load(f)


split=0.1
(X,y) = shuffle(X,y)
thres = int(len(y) * split)
(X,y) = shuffle(X,y)
Xval, yval = X[0:thres], y[0:thres]
Xtrain, ytrain = X[thres:], y[thres:]


model = Sequential()
model.add(Conv2D(16,(7,7),strides=(3,3),activation="relu", input_shape=(256,256,1)))
model.add(Dropout(0.1))
model.add(Conv2D(16,(3,3),strides=(1,1),activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=["accuracy"])

roc_curves = k_fold_crossvalidation(X,y,10,model,epochs=14)

def plot_crossval_auc(roc_curves):
    cmap = sns.cubehelix_palette(11)
    aucs=[]
    ax = plt.axes()
    for fold in roc_curves.keys():
        (f,p) = roc_curves[fold]
        aucs.append(area_under_curve(f,p))
        label_str = "fold {}, roc auc: {:.2f}".format(fold,-1*aucs[-1])
        ax.plot(f,p,label=label_str,color=cmap[fold])
    ax.plot([0,1],[0,1],label="random",color="black")
    ax.legend(loc="lower right")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for a basic convolutional neural net on 10 different validation folds")
    print("mean auc: {}, standard deviation: {}".format(np.mean(aucs),np.std(aucs)))
    plt.show()


for i in range(1,9):
    X[0:i*100],y[0:i*100]

