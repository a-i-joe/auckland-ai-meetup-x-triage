"""Plot saliency maps """
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.applications import inception_v3
from scipy.ndimage import imread, zoom
from scipy.misc import imresize
import matplotlib.pyplot as plt

def get_saliency(image,model):
    """Returns a saliency map with same shape as image. """
    K.set_learning_phase(0)
    K._LEARNING_PHASE = tf.constant(0)
    image = np.expand_dims(image,0)
    loss = K.variable(0.)
    loss += K.sum(K.square(model.output))
    grads = K.abs(K.gradients(loss,model.input)[0])
    saliency = K.max(grads,axis=3)
    fetch_saliency = K.function([model.input,K.learning_phase()],[loss,saliency])
    outputs, saliency = fetch_saliency([image,0])
    K.set_learning_phase(True)
    return saliency

def plot_saliency(image, model,ax):
    """Gets a saliency map for image, plots it next to image. """
    saliency = get_saliency(image,model)
    ax.imshow(np.squeeze(saliency),cmap="viridis")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.pause(0.01)

def get_gradcam(image,model,layer_name,mode):
    layer = model.get_layer(layer_name)
    image = np.expand_dims(image,0)
    loss = K.variable(0.)
    if mode == "abnormal":
        loss += K.sum(model.output)
    elif mode == "normal":
        loss += K.sum(1 - model.output)
    else:
        raise ValueError("mode must be normal or abnormal")
    #gradients of prediction wrt the conv layer of choice are used
    upstream_grads = K.gradients(loss,layer.output)[0]
    feature_weights = K.mean(upstream_grads,axis=[1,2]) #spatial global avg pool
    heatmap = K.relu(K.dot(layer.output, K.transpose(feature_weights)))
    fetch_heatmap = K.function([model.input, K.learning_phase()], [heatmap])
    return fetch_heatmap([image,0])[0]

def transparent_cmap(cmap,alpha,N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0,0.8,N+4)
    return mycmap

def plot_heatmap(image,model,layer_name,mode,ax,cmap=plt.cm.plasma,alpha=0.6):
    heat_cmap = transparent_cmap(cmap,alpha)
    heatmap = get_gradcam(image,model,layer_name,mode)
    image = np.squeeze(image)
    heatmap = np.squeeze(heatmap)
    heatmap = imresize(heatmap, image.shape)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.imshow(image,cmap="gray")
    ax.imshow(np.squeeze(heatmap),cmap=heat_cmap)
    ax.set_xticklabels([]); ax.set_yticklabels([])

def plot_network(image, model, label=None):
    layer_names = [l.name for l in model.layers if isinstance(l,Conv2D)]
    n_conv = len(layer_names)
    n_axes = n_conv 
    prediction = model.predict(np.expand_dims(image,0))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    fig, [axlist1, axlist2] = plt.subplots(2,n_conv)
    diagnosis = ["negative", "positive"]
    for j in range(n_conv):
        plot_heatmap(image, model, layer_names[j],"abnormal",axlist1[j])
#        axlist1[j].set_xlabel(layer_names[j] + "ab")
    for j in range(n_conv):
        plot_heatmap(image, model, layer_names[j],"normal",axlist2[j],cmap=plt.cm.inferno)
    fig.suptitle("Prediction: {},  {}".format(prediction,label))
    fig.show()
