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
- **Kernels**: A 3x3 matrix or grid of values that is passed over the input data, typically in a sliding manner. At each position, the kernel extracts local features or patterns from the input. These features can represent edges, corners, textures, or other relevant information depending on the specific task.
- **How do we decide the number of kernels?** There is no definitive rule for selecting the exact number of kernels. However based on the computational resources and the dataset size, the number of kernels can be determined.
- **Batch Normalization**: To normalize the activations of each layer by subtracting the batch mean and dividing by the batch standard deviation, reducing the internal covariate shift.
- **Image Normalization**: Refers to the process of adjusting the pixel values of an image to a standard range or distribution. It is commonly performed as a preprocessing step before feeding the images into a neural network.
- **DropOut**:
  - Dropout is used as a regularization technique in neural networks to prevent overfitting.
  - It helps in improving the generalization ability of the model by reducing co-adaptation among neurons.
  - Dropout introduces randomness by randomly dropping out a certain percentage of neurons or connections during training, which makes the network to learn more robust and diverse features.
- **When to use DropOut?**
  - Introduce dropout when you observe overfitting
  - Use dropout with large networks
