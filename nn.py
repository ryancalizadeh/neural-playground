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
    theta1: np.ndarray,
    theta2: np.ndarray
) -> np.ndarray:
    a0 = np.matmul(theta0, x)
    h0 = sig(a0)
    a1 = np.matmul(theta1, h0)
    h1 = sig(a1)
    a2 = np.matmul(theta2, h1)
    h2 = sig(a2)
    return h2

def cost(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray
) -> float:
    return 0.5 * np.sum((ff(x, theta0, theta1, theta2) - y) ** 2)

def get_grad(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray
) -> tuple:
    a0 = np.matmul(theta0, x)
    h0 = sig(a0)
    a1 = np.matmul(theta1, h0)
    h1 = sig(a1)
    a2 = np.matmul(theta2, h1)
    h2 = sig(a2)
    diff = h2 - y
    j = 0.5 * np.sum(diff ** 2)

    d1 = np.multiply(diff, sig_p(a2))
    d2 = np.multiply(np.matmul(theta2.T, d1), sig_p(a1))
    d3 = np.multiply(np.matmul(theta1.T, d2), sig_p(a0))

    theta2_grad = np.matmul(d1, h1.T)
    theta1_grad = np.matmul(d2, h0.T)
    theta0_grad = np.matmul(d3, x.T)

    return theta0_grad, theta1_grad, theta2_grad, j

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

    init_size = 1
    alpha = 0.008
    group_size = 100
    cycles = 2000

    theta0 = (np.random.random((32, 784)) - 0.5) * 2 * init_size
    theta1 = (np.random.random((16, 32)) - 0.5) * 2 * init_size
    theta2 = (np.random.random((10, 16)) - 0.5) * 2 * init_size

    # print(x.shape) (60000, 784)
    # print(y.shape) (60000, 10)

    theta0_grad_sum = np.zeros(theta0.shape)
    theta1_grad_sum = np.zeros(theta1.shape)
    theta2_grad_sum = np.zeros(theta2.shape)
    cost_sum = 0
    costs = []

    i = 0
    cycle = 0
    while cycle < cycles:

        if i % group_size == 0:
            theta0 -= alpha * theta0_grad_sum
            theta1 -= alpha * theta1_grad_sum
            theta2 -= alpha * theta2_grad_sum
            print(f"Cycle {cycle} avg cost: {cost_sum/group_size}")
            costs.append(cost_sum)
            
            theta0_grad_sum = np.zeros(theta0.shape)
            theta1_grad_sum = np.zeros(theta1.shape)
            theta2_grad_sum = np.zeros(theta2.shape)
            cost_sum = 0
            cycle += 1


        theta0_grad, theta1_grad, theta2_grad, j = get_grad(x[i:i+1, :].T, y[i:i+1, :].T, theta0, theta1, theta2)
        theta0_grad_sum += theta0_grad
        theta1_grad_sum += theta1_grad
        theta2_grad_sum += theta2_grad
        cost_sum += j

        i += 1
        if i >= n-10000:
            i = 0
    
    sample = 7687
    image = np.array(images[sample:sample+1, :])
    img = image.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    print(y[sample])
    print(ff(x[sample], theta0, theta1, theta2))

    plt.scatter(np.arange(len(costs)), costs)
    plt.show()
    
    print("Testing")
    cost_sum = 0
    for i in range(n-10000):
        cost_sum += cost(x[i:i+1, :].T, y[i:i+1, :].T, theta0, theta1, theta2)
    print("Training cost: ", cost_sum/(n-10000))

    cost_sum = 0
    for i in range(n-10000, n):
        cost_sum += cost(x[i:i+1, :].T, y[i:i+1, :].T, theta0, theta1, theta2)
    print("Testing cost: ", cost_sum/10000)

    print("Wrapping up")
    df = pd.DataFrame(theta0)
    json.to_json("theta0.json", df)
    df = pd.DataFrame(theta1)
    json.to_json("theta1.json", df)
    df = pd.DataFrame(theta2)
    json.to_json("theta2.json", df)


## TO DO ##
# Add comments
# Fstring printing
# Add Biases
# Allow custom architecture
# Add convolutions
# Try test dataset
# Look into regularization
# Fiddle with step size, epsilon, batch size, etc. 
# Check out:
#   Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron (2016). "6.5 Back-Propagation and Other Differentiation Algorithms". Deep Learning. MIT Press. pp. 200–220. ISBN 9780262035613.
    


if __name__=="__main__":
    main()