# neural-playground
Neural Network for handwritten digit classification trained with gradient descent and coded from scratch. The motivation behind this project was that I wanted to implement the math behind neural networks without relying on external libraries like Tensorflow or PyTorch. This led to a lot of interesting insights into the mechanics of backpropogation, and also gave me the freedom to interact directly with the structure of the network. For example, the folders imgs/ and imgs2/ contain visualizations of the weights from the input to individual nodes in the first layer after training.

![image](https://user-images.githubusercontent.com/31375351/149883703-cb1a3620-64f3-4282-87fb-ef610a24d6c9.png)

This is the visualization of the weights from the input layer to a node in the first layer of one of the models that was trained. Here, the brightness of a pixel in the 28x28 image represents the weight from the corresponding pixel in the input to the given node. You can make out a circle shape in the pattern here, which might suggest the network is conducting some crude feature extraction.
