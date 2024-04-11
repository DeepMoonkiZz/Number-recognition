import pandas as pd

from gradient_descent_gpu import *


data = pd.read_csv("/home/segal/Documents/Projets_perso/Number-recognition/dataset/train.csv")

data = cp.array(data)
m, n = data.shape
cp.random.shuffle(data)
data = data.T

Y = data[0].get()
X = data[1:n]
X = X / 255.
_, m = X.shape
