# Bike-Rental-Predictor
Based off a data set, using linear regression, this Neural Net will predict bike rentals.
It has one hidden layer and one output layer which use the sigmoid activation function.
NN propagates forward and back calculating the error to change the weights.

## Hyperparameters That Work for Me
```
iterations = 4000
learning_rate = 0.075
hidden_nodes = 10
output_nodes = 20
```
It seems that not much happens to the train vs validation loss on the last 500 iterations, so it may be unnecessary.
Graphs of the results are found in the results folder.
