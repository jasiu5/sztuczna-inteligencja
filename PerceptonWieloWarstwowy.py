import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class MultilayerPerceptron:
    def __init__(self, appr_func, num_hidden_neurons, seed, input_size=1) -> None:
        self.num_hidden_layers = len(num_hidden_neurons)
        self.num_hidden_neurons = num_hidden_neurons 
        self.input_size = input_size 
        self.init_weights_and_biases() 
        self.activation_f = self.logistic
        self.appr_func = appr_func
        np.random.seed(seed)
    
    @staticmethod
    def loss(y_pred, y_true):
        return (y_pred - y_true) ** 2

    @staticmethod
    def d_loss(y_pred, y_true):
        return 2 * (y_pred - y_true)

    def logistic(self, z):
        return 1 / (np.exp(-z) + 1)

    def d_logistic(self, z):
        return self.logistic(z) * (1 - self.logistic(z))

    def init_weights_and_biases(self):
        self.weights = [] 
        self.biases = []
        param = 1 / np.sqrt(self.input_size)
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.weights.append(np.random.uniform(-param, param, (self.num_hidden_neurons[i], self.input_size))) 
            else:
                self.weights.append(np.random.uniform(-param, param, (self.num_hidden_neurons[i], self.num_hidden_neurons[i-1]))) 
            self.biases.append(np.random.uniform(-param, param, (self.num_hidden_neurons[i], 1)))  
        self.weights.append(np.zeros((1, self.num_hidden_neurons[-1])))
        self.biases.append(np.zeros((1, 1))) 

                
    def feedforward(self, input):
        for i in range(self.num_hidden_layers):   
            layer_output = np.dot(self.weights[i], input) + self.biases[i]  
            layer_output_act = self.activation_f(layer_output)   
            input = layer_output_act
        exit_layer_output = np.dot(self.weights[-1], input) + self.biases[-1]         
        return exit_layer_output
        
    def train(self, training_data, epochs, mini_batch_size, learning_rate):
        for _ in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

    def update_mini_batch(self, mini_batch, learning_rate):
        """Aktualizuje wagi i biasy"""
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_weights, delta_biases = self.backpropagation(x, y)
            gradient_weights = [gw + dw for gw, dw in zip(gradient_weights, delta_weights)]
            gradient_biases = [gb + db for gb, db in zip(gradient_biases, delta_biases)]
        self.weights = [weight - gradient / len(mini_batch) * learning_rate for weight, gradient in zip(self.weights, gradient_weights)]
        self.biases = [bias - gradient / len(mini_batch) * learning_rate for bias, gradient in zip(self.biases, gradient_biases)]

    def backpropagation(self, x, y):
        """Zwraca krotke wag i biasow reprezentujaca gradient dla funckji bledu"""
        weights = [np.zeros(w.shape) for w in self.weights]
        biases = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        vectors = []
        for w, b in zip(self.weights, self.biases):
            v = np.dot(w, activation) + b
            vectors.append(v)
            activation = self.logistic(v)
            activations.append(activation)
        delta = self.d_loss(vectors[-1], y)
        biases[-1] = delta
        weights[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_hidden_layers + 2):
            v = vectors[-i]
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * self.d_logistic(v)
            weights[-i] = np.dot(delta, activations[-i - 1].transpose())
            biases[-i] = delta
        return (weights, biases)

    def predict(self, test_inputs):
        y_predicted = []
        for x in test_inputs:
            y_predicted.append(self.feedforward(x)[0])
        return y_predicted

def f(x):
    return x**2 * np.sin(x) + 100*np.sin(x) * np.cos(x)

def main():
    network = MultilayerPerceptron(appr_func=f, num_hidden_neurons=[25, 25], seed=10)
    X = np.linspace(-15, 15, 5000)
    Y = np.array([f(x) for x in X])
    Y = Y.reshape((len(Y), 1))
    scale_y = MinMaxScaler()
    Y = scale_y.fit_transform(Y)
    Y = Y.reshape((len(Y)))
    network.train(training_data=[(x, y) for x, y in zip(X, Y)], epochs=10000, mini_batch_size=100, learning_rate=0.35)
    Y_predict = np.array(network.predict(X))
    Y = Y.reshape((len(Y), 1))
    Y_predict = Y_predict.reshape((len(Y_predict), 1))
    Y= scale_y.inverse_transform(Y)
    Y_predict= scale_y.inverse_transform(Y_predict)
    plt.plot(X, Y, label='Actual')
    plt.plot(X, Y_predict, label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

