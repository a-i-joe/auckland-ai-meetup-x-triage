import numpy as np
import util
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
import dill as pickle
from sklearn.utils import shuffle
import argparse

save_path = "test.pickle"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--datapath", help="path to the pickled data", default=save_path)
args = parser.parse_args()
data_path = args.datapath

with open(data_path, "r") as f:
    (X, y, info) = pickle.load(f)
(X, y) = shuffle(X, y)
s = 720
Xtr, ytr = X[:s], y[:s]
Xt, yt = X[s:], y[s:]

input_shape = X.shape[1:]

model = Sequential()
model.add(Conv2D(16, (7, 7), strides=(3, 3),
                 activation="relu", input_shape=input_shape))
model.add(Dropout(0.1))
model.add(Conv2D(16, (3, 3), strides=(1, 1), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer=Adam(), metrics=["accuracy"])


roc_curves, safeset_percents = util.k_fold_crossvalidation(
    X, y, 10, model, epochs=14)
aucs = [util.area_under_curve(f, p) for (f, p) in roc_curves.values()]
print("Got average safeset % of {:.2f}. min :{:.2f},"
      " max: {:.2f}, std: {:.2f}".format(
        np.mean(safeset_percents), min(safeset_percents),
        max(safeset_percents), np.std(safeset_percents)))
print("Got average ROC AUC of {:.2f}. min :{:.2f},"
      " max: {:.2f}, std: {:.2f}".format(
        np.mean(aucs), min(aucs), max(aucs), np.std(aucs)))
util.plot_crossval_auc(roc_curves)
