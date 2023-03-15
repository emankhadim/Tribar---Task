# -----------------------------------------------------------------------------------------------------------------------
                                    # EMAN KHADIM
                                    # eman.khadim@fau.de
                                    # 01782080442
                                    # AI Software Developer
#________________________________________________________________________________________________________________________
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
#------------------------------------------------------------------------------------------------------------------------

def xor(a, b):
    return (a and not b) or (not a and b)


def toggle_switch(a, b):
    return a != b


def neural_network(neurons, hidden_layers, lr, epochs, x_train, y_train, x_test, y_test, prob):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu', input_shape=(2,)))  # input layer
    for i in range(hidden_layers):  # hidden layers
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output layer
    model.summary()
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    # train the model
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)

    if prob == "xor":
        print("--------------------------------------------------------------------------")
        print(f'\t\t\tTest accuracy of XOR function: {test_acc}')
    else:
        print("--------------------------------------------------------------------------")
        print(f'\t\t\tTest accuracy of Toggle Switch Circuit: {test_acc}')
    return test_acc


def check_threshold(threshold, accuracy):
    if accuracy > threshold:
        print("\t\t\tModel is accurate enough to be selected.")
        print("--------------------------------------------------------------------------")
    else:
        print("\t\t\tModel is not accurate enough to be selected.")
        print("--------------------------------------------------------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-problem", type=str, choices=["xor", "toggle"], default="xor", help="Problem to solve")
    parser.add_argument("-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("-num_hidden_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-num_neurons", type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument("-learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-threshold", type=float, default=0.95, help="Accuracy threshold")
    args = parser.parse_args()

    if args.problem == "xor":
        # Generating data for xor
        train_x = np.array([[random.randint(0, 1), random.randint(0, 1)] for _ in range(args.samples)])
        train_y = np.array([xor(a, b) for a, b in train_x])
        test_x = np.array([[random.randint(0, 1), random.randint(0, 1)] for _ in range(args.samples)])
        test_y = np.array([xor(a, b) for a, b in test_x])
        accuracy = neural_network(args.num_neurons, args.num_hidden_layers, args.learning_rate, args.num_epochs,train_x, train_y, test_x, test_y, args.problem)
    elif args.problem == "toggle":
        # Generating data for toggle
        train_x = np.array([[random.randint(0, 1), random.randint(0, 1)] for _ in range(args.samples)])
        train_y = np.array([toggle_switch(a, b) for a, b in train_x])
        test_x = np.array([[random.randint(0, 1), random.randint(0, 1)] for _ in range(args.samples)])
        test_y = np.array([toggle_switch(a, b) for a, b in test_x])
        accuracy = neural_network(args.num_neurons, args.num_hidden_layers, args.learning_rate, args.num_epochs,train_x, train_y, test_x, test_y, args.problem)

    check_threshold(args.threshold, accuracy)
