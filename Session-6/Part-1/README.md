# Backpropagation in Neural Networks

This README.md file provides an overview of the backpropagation algorithm in neural networks, explaining its purpose and how it works.

#### How to read the excel file present in this repository?
The Excel file has `9` sheets in total:
- **Neural Network**: A Neural Network Architecture, with 
    - Input Layer 2 Neurons
    - 1 Hidden Layer with 2 Neurons
    - 1 Output Layer with 2 Neurons
- **Terminologies**: All the Variable names are explained. 
    - `i` input neuron
    - `w` weights
    - `o` output neuron
    - `h` neurons in the hidden layer
    - `a_` output of the activation functions
    - `σ` activation function
    - `t` expected target
    - `E` loss function
- **Calculations**: Backpropagation Calculation is done here. It has the screenshots of the manually calculated partial derivatives. <Under Construction>
- **LR_0.1**, **LR_0.2**, **LR_0.5**, **LR_0.8**, **LR_1.0**, **LR_2.0**: The performance of the model is calculated for the different Learning Rates. In the upcoming sections, the model's performance is described using visuals.

#### Let's understand what happens during the Forward Propagation
  - Initialize the weights and biases randomly
  - Calculate the weighted sum of inputs at the hidden layer neurons and apply an activation function (e.g., sigmoid or ReLU) to obtain the hidden layer activations.
  - Repeat the same process for the output layer, using the hidden layer activations as inputs.

#### Backpropagation:
- Calculate the error between the predicted output and the desired output.
- Compute the gradients of the loss function with respect to the weights and biases in the output layer.
- Propagate these gradients backward to the hidden layer, calculating the gradients at the hidden layer neurons.
- Finally, update the weights and biases in both the hidden and output layers using an optimization algorithm (e.g., gradient descent) and the calculated gradients.

**Repeat**:
Repeat the forward propagation and backpropagation steps for multiple iterations or until the network converges.

#### Backpropagation Calculations Explanation with an example:
The step-by-step calculation for adjusting the weights using partial derivatives in a neural network with two neurons in the input layer, two neurons in the hidden layer, and two neurons in the output layer.

  Let's denote the following variables:
  - Inputs: i1, i2
  - Weights: w1, w2, w3, w4, w5, w6, w7, w8
  - Neurons of hidden layers: h1, h2
  - Outputs: o1, o2
  - Expected Targets: t1, t2
  - Error functions: E1, E2
  - Activation function: σ(x) = 1 / (1 + exp(-x))
  
  Formulas:
  - E1 = ½ * (t1 - o1)²
  - E2 = ½ * (t2 - o2)²
  - E_total = E1 + E2
  - h1 = w1i1 + w2i2
  - h2 = w3i1 + w4i2
  - a_h1 = σ(h1) = 1 / (1 + exp(-h1))
  - a_h2 = σ(h2) = 1 / (1 + exp(-h2))
  - o1 = w5a_h1 + w6a_h2
  - o2 = w7a_h1 + w8a_h2
