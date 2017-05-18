import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

def plot_images(X,s=5):
    """plot s**2 images in a square grid of
        length s from ndarray of images X. """
    (n,w,h,c)=X.shape
    ret=np.zeros((w*s,h*s,c))
    for x in range(s):
        for y in range(s):
            n=np.random.randint(low=0,high=X.shape[0])
            ret[x*w:x*w+w,y*h:y*h+h,:] = X[n]
    print(ret.shape)
    plt.imshow(ret.reshape(ret.shape[0],ret.shape[0]));plt.show()

def get_true_pos_rate(predictions,targets):
    """True positive rate/sensitivity"""
    n_true_pos =( predictions * targets).sum()
    n_pos = targets.sum()
    return n_true_pos/float(n_pos)

def get_false_pos_rate(predictions,targets):
    """False positive rate/specificity """
    n_false_pos =( predictions * (1 - targets)).sum()
    n_neg = (1.0 - targets).sum()
    return n_false_pos / float(n_neg)

def roc_curve(outputs,targets,n=100):
    """Computes ROC by moving a threshold """
    targets = targets.astype("int16")
    thresholds = np.linspace(1,0,n)
    falsepos, truepos = [], []
    for t in thresholds:
        predictions = (outputs>t).astype("int16")
        false_pos_rate = get_false_pos_rate(predictions,targets)
        true_pos_rate = get_true_pos_rate(predictions,targets)
        falsepos.append(false_pos_rate)
        truepos.append(true_pos_rate)
    return falsepos,truepos


def area_under_curve(x,y):
    """Finds the area under unevenly spaced curve y=f(x)
        using the trapezoid rule. x,y should be arrays of reals with same length."""
    a = 0.0
    for i in range(0,len(x)-1):
        a +=  (x[i+1] - x[i]) * (y[i+1] + y[i])
    a = a * 0.5
    return a

def get_roc_auc(X,y,model):
    outputs = model.predict(X).reshape(y.shape)
    falsepos,truepos = roc_curve(outputs,targets)
    a = area_under_curve(falsepos,truepos)
    return a

def safeset_percent(predictions,targets,tolerance=1,dx=0.001):
    """Return the % of normal xrays that can be filtered out
        without the % of false negatives surpassing tolerance."""
    threshold = 0
    tolerance_exceeded = False
    n_normal = float((1-targets).sum())
    n_abnormal = float(len(targets) - n_normal)
    while not tolerance_exceeded:
        above_threshold = (predictions>threshold).astype("int16")
        false_negatives = ((1-above_threshold) * targets).sum() #where predictions=0 and targets=1
        if false_negatives >= tolerance:
            tolerance_exceeded = True
        n_true_negatives = ((1-above_threshold) * (1-targets)).sum()#where predictions=0 and targets=0
        threshold += dx
    return 100 * n_true_negatives / n_normal

def get_safeset(X,y,model,tolerance=1):
    outputs = model.predict(X).reshape(y.shape)
    return safeset_percent(outputs,y,tolerance)

def k_fold_crossvalidation(X,y,k,model,epochs=10,tmp_path="/tmp/params.h5"):
    """Splits data into n (train, val) folds and trains model on each one. 
        Model must be a (compiled) instance of the keras model or sequential class,"""
    print("running {} fold cross validation...".format(k))
    model.save_weights(tmp_path)
    n_examples = X.shape[0]
    w = int(n_examples / k )
    roc_curves = {}
    safeset_percents = []
    for fold  in range(k):
        model.load_weights(tmp_path)
        val_start = w * fold 
        val_end = val_start + w
        Xval = X[val_start:val_end]
        yval = y[val_start:val_end]
        Xtrain = np.concatenate([X[val_end:], X[:val_start]])
        ytrain = np.concatenate([y[val_end:], y[:val_start]])
        hist = model.fit(Xtrain,ytrain,epochs=epochs,validation_data=(Xval,yval))
        outputs = model.predict(Xval).reshape(yval.shape)
        (falsepos,truepos) = roc_curve(outputs,yval)
        roc_curves[fold] = (falsepos,truepos)
        safeset_percents.append(get_safeset(Xval,yval,model,1))
    return roc_curves, safeset_percents


def plot_crossval_auc(roc_curves):
    cmap = sns.cubehelix_palette(11)
    aucs=[]
    ax = plt.axes()
    for fold in roc_curves.keys():
        (f,p) = roc_curves[fold]
        aucs.append(area_under_curve(f,p))
        label_str = "fold {}, roc auc: {:.2f}".format(fold,aucs[-1])
        ax.plot(f,p,label=label_str,color=cmap[fold])
    ax.plot([0,1],[0,1],label="random, roc auc: 0.5",color="black")
    ax.legend(loc="lower right")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves across 10 different validation folds(tiny convnet trained on small datasets)")
    plt.show()
