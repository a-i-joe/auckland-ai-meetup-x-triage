import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def plot_images(X,s=5):
    """plot s**2 images in a square grid of
        length s from ndarray of images X"""
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
    n_false_pos =( predictions * (1-targets)).sum()
    n_neg = (1.0-targets).sum()
    return n_false_pos / float(n_neg)

def roc_curve(outputs,targets,n=100):
    """ """
    targets = targets.astype("int16")
    thresholds = np.linspace(0,1,n)
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
        using the trapezoid rule. x,y should be arrays of reals withthe same length."""
    a = 0.0
    for i in range(0,len(x)-1):
        a +=  (x[i+1] - x[i]) * (y[i+1] + y[i])
    a = a * 0.5
    return a

def get_roc_auc(X,y,model):
    model.predict(X).reshape(y.shape).astype("int16")
    falsepos,truepos = roc_curve(outputs,targets)
    a = area_under_curve(falsepos,truepos)
    return a

def plot_roc_curve(ax,falsepos,truepos):
    print("plotting roc")
    ax.plot(falsepos,truepos)
    ax.plot(falsepos,falsepos)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    
def show_roc_auc(X,y,model,n=150):
    outputs = model.predict(X).reshape(y.shape)
    falsepos,truepos = roc_curve(outputs,y)
    auc = area_under_curve(falsepos,truepos)
    ax = plt.axes()
    plot_roc_curve(ax,falsepos,truepos)
    print("curve 1, auc:"+str(auc))
    plt.show()

def k_fold_crossvalidation(X,y,k,model,epochs=10,tmp_path="/tmp/params.h5"):
    """Splits data into n (train, val) folds and trains model on each one. 
        Model must be a (compiled) instance of the keras model or sequential class,"""
    model.save_weights(tmp_path)
    n_examples = X.shape[0]
    w = int(n_examples / k )
    roc_curves = {}
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
    return roc_curves
