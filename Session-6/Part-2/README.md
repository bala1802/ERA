#### Objective

The purpose of this repository is to achieve `99.4%` validation accuracy on the [MNIST Dataset]([url](https://www.tensorflow.org/datasets/catalog/mnist)). 

**Conditions**
- The Network should hold Lesser than 20K parameters
- The Training module should be lesser than 20 epochs

#### Definitions

- **Max Pooling**: To reduce the spatial dimensions of the input while retaining the most important features by selecting the maximum value within each pooling region.
- **1x1 Convolutions**: To perform dimensionality reduction and feature transformation while preserving spatial information.
- **3x3 Convolutions**: Refers to the region of the input data that a particular feature or neuron in the network sees or takes into account when making predictions.
- **SoftMax**: `softmax(x) = exp(x) / sum(exp(x))` Where `exp(x)` is the exponential function applied element-wise to the logits, and the division by the sum ensures the resulting probabilities sum up to 1. By using softmax, the neural network outputs a probability distribution that can be used to make predictions or evaluate the confidence of the model for different classes. The class with the highest probability is often considered as the final prediction.
- **Learning Rate**: To control the size of the updates made to the model's parameters during training, influencing the speed and stability of the learning process.
