# Objective

The purpose of creating this repository is to achieve 70% Test accuracy for the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) based on the below constraints,

- Build 3 different models with `Group Normalization` or `Layer Normalization` or `Batch Normalization`
- The number of epochs should be less than or equal to `20`
- The total parameters used in the Neural Network architecture must be less than or equal to `50K`

# How to read this repository?
- `model.py` is a helper script, where the various Model architectures are constructed.
- `util.py` is a helper script, where the data utility is handled, also the model related hyper parameters are configured, including the model training
- The `Session-8.ipynb` is the main notebook. The Constructed Models are initialized and sent to `util.py` for the model training.

# Observation of each Model
## Model-1
- This model is experimented with `Group Normalization`.
- Achieved ~70% accuracy in less than 20 epochs.

### Training vs Test (Accuracy and Losses)
![image](https://github.com/bala1802/ERA/assets/22103095/30f43f97-f490-4c6d-9346-a5fa5ce88737)

## Model-2
- This model is experimented with `Layer Normalization`.
- Achieved ~70% accuracy in less than 20 eochs

### Training vs Test (Accuracy and Losses)
![image](https://github.com/bala1802/ERA/assets/22103095/e81ea7a2-942a-49d1-bab5-2b99153eb4b8)

## Model-3
- This model is experimented with `Batch Normalization`.
- Achieved ~70% accuracy in less than 20 eochs

### Training vs Test (Accuracy and Losses)
![image](https://github.com/bala1802/ERA/assets/22103095/27ce8403-3886-4933-8157-5b8c39ace93a)


