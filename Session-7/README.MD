# Objective
The purpose of creating this repository is to achieve `99.4%` Test accuracy for the MNIST Dataset (http://yann.lecun.com/exdb/mnist/) based on the below constraints
- The Neural Network architecture should hold less than 8K Parameters
- The number of epochs the model trained should be <= 15 Epochs

# How to read this repository?
- `model.py` is a helper script, where the various Model architectures are constructed.
- `util.py` is a helper script, where the data utility is handled, also the model related hyper parameters are configured, including the model training
- The `Session-7.ipynb` is the main notebook. The Constructed Models are initialized and sent to `util.py` for the model training.


