# Neural-Network

### Project: Back-Propagation for Handwritten Digit Recognition

##### Introduction
This is a program that builds a **2-layer, feed-forward neural network** and trains it using the __back-propagation algorithm__. The problem that the neural network will handle is a multi-class classification problem for recognizing images of handwritten digits.

##### Algorithm
1) All inputs to the neural network will be numeric. 
2) The neural network has one hidden layer. 
3) The network is fully connected between consecutive layers, meaning each unit, which we’ll call a node, in the input layer is connected to all nodes in the hidden layer, and each node in the hidden layer is connected to all nodes in the output layer. 
4) Each node in the hidden layer and the output layer will also have an extra input from a “bias node" that has constant value +1. So, we can consider both the input layer and the hidden layer as containing one additional node called a bias node. 
5) All nodes in the hidden layer (except for the bias node) should use the __ReLU activation function__, while all the nodes in the output layer should use the __Softmax activation function__. 
6) The initial weights of the network will be set __randomly__ based on an input random seed (already implemented in the skeleton code). 
7) Assuming that input examples (called instances in the code) have m attributes (hence there are m input nodes, not counting the bias node) and we want h nodes (not counting the bias node) in the hidden layer, and o nodes in the output layer, then the total number of weights in the network is (m+1)h between the input and hidden layers, and (h+1)o connecting the hidden and output layers. The number of nodes to be used in the hidden layer will be given as input. 


