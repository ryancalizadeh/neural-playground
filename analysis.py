print("Loading Modules")
from os import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io import json

def read_json(fn: str) -> np.ndarray:
    ret = json.read_json(fn)
    ret = ret.to_numpy()
    return ret

def main():
    print("Reading Files")
    theta0 = read_json("theta0.json")
    theta1 = read_json("theta1.json")

    print("Conducting Analysis")
    for i in range(16):
        img: np.ndarray = theta0[i:i+1, :]
        img = img.reshape((28, 28))
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.savefig(fname="img" + str(i) + ".png")


if __name__=="__main__":
    main()