Development environment used : Python 3.8 , TensorFlow 2.3 , Libraries used: Numpy , Argparse , Random.

How to run this code through the command line.

Assuming you have saved the code in a file named neural_network.py, you can follow these steps:

Open the terminal or command prompt on your computer.

Navigate to the directory where the neural_network.py file is located using the cd command.

1. To run the program for the XOR problem with default settings, type the following command and press enter:

                       python neural_network.py -problem xor
		       
2. To run the program for the toggle switch problem with default settings, type the following command and press enter:

                       python neural_network.py -problem toggle
		       
3. You can also specify additional command line arguments to customize the behavior of the program. For example, to set the number of hidden layers to 2 and the    number of neurons in each layer to 16, you can use the following command:

                       python neural_network.py -problem xor -num_hidden_layers 2 -num_neurons 16
		       
4. To see a list of all available command line arguments and their default values, you can use the --help option:

                       python neural_network.py --help

