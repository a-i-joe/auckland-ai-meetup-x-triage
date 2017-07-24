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
    fetch_saliency = K.function([model.input],[loss,saliency])
    outputs, saliency = fetch_saliency([image])
    K.set_learning_phase(True)
    return saliency

def plot_saliency(image, model):
    """Gets a saliency map for image, plots it next to image. """
    saliency = get_saliency(image,model)
    plt.ion()
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.imshow(np.squeeze(saliency),cmap="viridis")
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax2.imshow(np.squeeze(image),cmap="gray")
    ax2.set_xticklabels([]); ax2.set_yticklabels([])
    plt.pause(0.01)
    plt.show()

def get_gradcam(image,model,layer_name):
    #remove dropout/noise layers
    K.set_learning_phase(0)
    K._LEARNING_PHASE = tf.constant(0)
    layer = model.get_layer(layer_name)
    image = np.expand_dims(image,0)
    loss = K.variable(0.)
    loss += K.sum(model.output)
    #gradients of prediction wrt the conv layer of choice are used
    upstream_grads = K.gradients(loss,layer.output)[0]
    feature_weights = K.mean(upstream_grads,axis=[1,2])
    heatmap = K.relu(K.dot(layer.output, K.transpose(feature_weights)))
    fetch_heatmap = K.function([model.input], [heatmap])
    return fetch_heatmap([image])[0]


def transparent_cmap(cmap,alpha,N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0,0.8,N+4)
    return mycmap


def plot_heatmap(image,model,layer_name,ax,cmap=plt.cm.jet,alpha=0.6):
    heat_cmap = transparent_cmap(cmap,alpha)
    heatmap = get_gradcam(image,model,layer_name)
    image = np.squeeze(image)
    heatmap = np.squeeze(heatmap)
    heatmap = imresize(heatmap, image.shape)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.imshow(image,cmap="gray")
    ax.imshow(np.squeeze(heatmap),cmap=heat_cmap)
    ax.set_xticklabels([]); ax.set_yticklabels([])

def plot_network(images, model, labels=None):
    layer_names = [l.name for l in model.layers if isinstance(l,Conv2D)]
    n_conv = len(layer_names)
    n_images = images.shape[0]
    fig, axlist = plt.subplots(n_images,n_conv)
    diagnosis = ["negative", "positive"]
    for i in range(n_images):
        for j in range(n_conv):
            plot_heatmap(images[i], model, layer_names[j], axlist[i][j])
            axlist[-1][j].set_xlabel(layer_names[j])
        if labels is not None:
            axlist[i][0].set_ylabel(str(labels[i]))
    fig.show()

