# Objective
The purpose of creating this repository is to achieve `99.4%` Test accuracy for the MNIST Dataset (http://yann.lecun.com/exdb/mnist/) based on the below constraints
- The Neural Network architecture should hold less than 8K Parameters
- The number of epochs the model trained should be <= 15 Epochs

# How to read this repository?
- `model.py` is a helper script, where the various Model architectures are constructed.
- `util.py` is a helper script, where the data utility is handled, also the model related hyper parameters are configured, including the model training
- The `Session-7.ipynb` is the main notebook. The Constructed Models are initialized and sent to `util.py` for the model training.

# Observation of each model
## Model-1
- This model is built for setting up the architecture.
- ~6.3M parameters.
- Unable to achieve the test accuracy to ~ `99.4%`.
- The best training accuracy is `99.96%` and the best test accuracy is `99.25%`
- This model is overfitting, as you can see improving the training accuracy from `99.96`% is not going to help converting the test accuracy `99.4`%
## Model-2
Let's build a model by reducing the number of channels at the end layers
- The model parameter is reduced to ~ 700K
- The channels in the end layers of the model is reduced.
- The training accuracy at Epoch-14 is `99.87` and the test accuracy is `99.13%`. The model is definitely overfitting
## Model-3
- The number of parameters are reduced to ~48K Parameters
- In the end layers, the channels are reduced from 64->32->16
- Then the channels are converted to the number of labels 10
- Still the model is overfitting
## Model-4
- The number of parameters used are same as the Model-3
- The paddings are removed from the initial layers and added in the middle convolutional layers
- In total 3 max pooling layers are added
- the 3rd maxpooling layer is added just before the prediction layer
## Model-5
- The total number of parameters are reduced to ~10.7K
- The final layer is convolved using 7X7 instead 3X3 and the receptive field is achieved
- The model is not overfitting, training further can improve the test accuracy.
- The Best Training accuracy is `98.6` and the best Test accuracy is `98.23`
## Model-6
- Added `BatchNormalization` to all the convolutional layers except the last layer.
- The model is overfitting.
- The best training accuracy is `99.72` and test accuracy is `98.99%`
## Model-7
 - Added Dropouts, regularization technique
 - No change in the number of parameters
 - The best training accuracy is 99.22% and the best test accuracy is 99.15%




