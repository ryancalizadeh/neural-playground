print("Importing")
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io import json

def sig(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def sig_p(z: np.ndarray) -> np.ndarray:
    s = sig(z)
    return np.multiply(s, 1 - s)


def ff(
    x: np.ndarray,
    theta0: np.ndarray,
    theta1: np.ndarray
) -> np.ndarray:
    a0 = np.matmul(theta0, x)
    h0 = sig(a0)
    a1 = np.matmul(theta1, h0)
    h1 = sig(a1)
    return h1

def cost(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    theta1: np.ndarray
) -> float:
    return 0.5 * np.sum((ff(x, theta0, theta1) - y) ** 2)

def get_grad(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    theta1: np.ndarray
) -> tuple:
    a0 = np.matmul(theta0, x)
    h0 = sig(a0)
    a1 = np.matmul(theta1, h0)
    h1 = sig(a1)
    diff = h1 - y
    j = 0.5 * np.sum(diff ** 2)

    prod = np.multiply(diff, sig_p(a1))

    theta1_grad = np.matmul(prod, h0.T)
    theta0_grad = np.matmul(np.multiply(np.matmul(theta1.T, prod), sig_p(a0)), x.T)

    return theta0_grad, theta1_grad, j

def main():
    print("Preprocessing")
    mndata = MNIST('samples')

    images, labels = mndata.load_training()
    images = np.array(images)
    x = np.array(images, dtype=np.float64) / 255
    n = x.shape[0]

    labels = np.array([labels]).T
    
    y = np.equal(np.arange(10), np.repeat(labels, 10, 1))

    print("Training")

    init_size = 0.5
    alpha = 0.005
    group_size = 3000
    cycles = 175

    theta0 = (np.random.random((16, 784)) - 0.5) * 2 * init_size
    theta1 = (np.random.random((10, 16)) - 0.5) * 2 * init_size

    print(x.shape) # (60000, 784)
    print(y.shape) # (60000, 10)

    theta0_grad_sum = np.zeros(theta0.shape)
    theta1_grad_sum = np.zeros(theta1.shape)
    cost_sum = 0
    costs = []

    print(x[5:6, :].shape)
    print(y[5:6, :].shape)

    i = 0
    cycle = 0
    while cycle < cycles:

        if i % group_size == 0:
            theta0 -= alpha * theta0_grad_sum
            theta1 -= alpha * theta1_grad_sum
            print("avg cost: ", cost_sum/group_size)
            costs.append(cost_sum)
            
            theta0_grad_sum = np.zeros(theta0.shape)
            theta1_grad_sum = np.zeros(theta1.shape)
            cost_sum = 0
            cycle += 1


        theta0_grad, theta1_grad, j = get_grad(x[i:i+1, :].T, y[i:i+1, :].T, theta0, theta1)
        theta0_grad_sum += theta0_grad
        theta1_grad_sum += theta1_grad
        cost_sum += j

        i += 1
        if i == n:
            i = 0
    
    sample = 7687
    image = np.array(images[sample:sample+1, :])
    img = image.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    print(y[sample])
    print(ff(x[sample], theta0, theta1))

    plt.scatter(np.arange(len(costs)), costs)
    plt.show()
    
    print("Wrapping up")
    df = pd.DataFrame(theta0)
    json.to_json("theta0.json", df)
    df = pd.DataFrame(theta1)
    json.to_json("theta1.json", df)


## TO DO ##
# Add comments
# Add Biases
# Add layer(s)
# Add convolutions
# Try test dataset
# Look into regularization
# Fiddle with step size, epsilon, batch size, etc. 
    


if __name__=="__main__":
    main()