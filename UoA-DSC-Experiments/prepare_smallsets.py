from prepro import basic_load
import numpy as np
import dill as pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--res", nargs="+", type=int,
                    help="desired reolution of images", default=[256, 256])
parser.add_argument("--dir", nargs="+", help="directories of data")
parser.add_argument(
    "--savepath", help="Path of file to store processed data in")

args = parser.parse_args()
res = tuple(args.res)
directories = args.dir

save_path = args.savepath
X, y, info = [], [], []
for d in directories:
    X_curr, y_curr, info_curr = basic_load(d, res)
    X.append(X_curr)
    y.append(y_curr)
    info += info_curr
X = np.concatenate(X)
X = np.expand_dims(X, axis=-1)
y = np.concatenate(y)

with open(save_path, "wb") as f:
    pickle.dump((X, y, info), f)
