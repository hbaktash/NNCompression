import math

import os
import random
import sys
import numpy as np

sys.path.append('/path/to/pythonExperiment')
from pythonExperiment import mnist_loader

import time


# Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class HashedNNetwork(object):

    def __init__(self, sizes, reduction_rate):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """ A list of buckets is assigned for every weight matrix. Size is reduced by the reduction_rate"""
        self.weight_buckets = [np.random.randn(int(math.ceil((x * y) / reduction_rate))) for x, y in
                               zip(sizes[:-1], sizes[1:])]

        for wight_buckets in self.weight_buckets:
            print("buckets size:", len(wight_buckets))

        """this is a pseudo implementation of the 
        hashing trick which does not reduce memory 
        usage and can only be used to determine accuracy of the HashNet trick"""
        self.weight_pointer_matrices = [(np.random.rand(nex, pre) * len(weight_bucket_set)).astype(int)
                                        for pre, nex, weight_bucket_set in
                                        zip(sizes[:-1], sizes[1:], self.weight_buckets)]
        self.weight_hash_signs = [(np.round(np.random.rand(nex, pre)) * 2) - 1 for pre, nex in
                                  zip(sizes[:-1], sizes[1:])]
        self.bucket_to_matrix_pointers = [[[] for _ in range(len(weight_bucket_set))] for weight_bucket_set in
                                          self.weight_buckets]
        for l in range(self.num_layers - 1):
            for i in range(self.sizes[l + 1]):
                for j in range(self.sizes[l]):
                    self.bucket_to_matrix_pointers[l][self.hash_function(l, i, j)].append(
                        (i, j, self.weight_hash_signs[l][i, j]))
        self.weights = [
            np.array([[self.get_weight(l, i, j) for j in range(self.sizes[l])] for i in range(self.sizes[l + 1])]) for l
            in range(self.num_layers - 1)]

    """ 
    hash functions --> (layer_index, i ,j)
        ------
        These function resemble the same weight_matrix indexing, i.e 'i' is 
        the node index in the next layer and 'j' is the node index in the previous layer.
    """

    def hash_function(self, layer_index, next_layer_index, prev_layer_index):
        # returns the bucket index
        return self.weight_pointer_matrices[layer_index][next_layer_index, prev_layer_index]

    def get_weight(self, layer_index, next_layer_index, prev_layer_index):
        # returns the weight of the index
        return self.weight_buckets[layer_index][self.hash_function(layer_index, next_layer_index, prev_layer_index)]

    def hash_hit(self, layer_index, next_layer_index, prev_layer_index, expected_index):
        return bool(self.hash_function(layer_index, next_layer_index, prev_layer_index) == expected_index)

    def get_delta_nabla_buckets_from_nabla_weights(self, delta_nabla_w):
        return [
            [sum([(nabla_w[pointer[0], pointer[1]] * pointer[2]) for pointer in pointers]) for pointers
             in
             bucket_to_matrix_pointer_set] for bucket_to_matrix_pointer_set, nabla_w in
            zip(self.bucket_to_matrix_pointers, delta_nabla_w)]

    def update_weights_from_buckets(self):
        for weight_bucket_set, buckets_to_matrix_pointers_set, w in zip(self.weight_buckets,
                                                                        self.bucket_to_matrix_pointers,
                                                                        self.weights):
            for pointers, weight_bucket in zip(buckets_to_matrix_pointers_set, weight_bucket_set):
                for pointer in pointers:
                    w[pointer[0], pointer[1]] = weight_bucket * pointer[2]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            cnt = 0
            for mini_batch in mini_batches:
                cnt += 1
                print("     {0}/{1} batch".format(cnt, len(mini_batches)))
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("##### Epoch {0}: {1} / {2} #####".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_weight_buckets = [np.zeros(buckets.shape) for buckets in self.weight_buckets]
        t0 = time.time()
        for x, y in mini_batch:
            """loop implementation"""
            # delta_nabla_b, delta_nabla_weight_buckets = self.backprop(x, y)
            """matrix BP implementat   ion"""
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # t1 = time.time()
            delta_nabla_weight_buckets = self.get_delta_nabla_buckets_from_nabla_weights(delta_nabla_w)
            # print("updating buckets time:", time.time() - t1)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_weight_buckets = [nw + dnw for nw, dnw in zip(nabla_weight_buckets, delta_nabla_weight_buckets)]
        self.weight_buckets = [w - (eta / len(mini_batch)) * nw
                               for w, nw in zip(self.weight_buckets, nabla_weight_buckets)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        # t1 = time.time()
        self.update_weights_from_buckets()
        # print("weights time:", time.time() - t1)
        print("batch time:", time.time() - t0)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        """forward matrix implementation"""
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        """backward matrix implementation"""
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            np.dot(self.weights[-l + 1].transpose(), delta)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def main(min_layers, max_layers, reduction_rate):
    # load data in [(x,y), ...] format
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("len test:", len(test_data))
    accuracies = []
    with open(os.path.expanduser("~/brainside/pythonExperiment/hashed_one_layer_test.txt"), 'w+') as f:
        f.write("reduction rate: 1/{}\n".format(reduction_rate))
    for mid_layer_size in range(min_layers, max_layers):
        print("######"
              "Training Hashed Net with {} hidden layers and 1/{} reduction\n".format(str(mid_layer_size),
                                                                                      str(reduction_rate)))
        net = HashedNNetwork([28 * 28, mid_layer_size, 10], reduction_rate)
        net.SGD(training_data, 100, 50, eta=5, test_data=test_data)
        accuracy = net.evaluate(test_data) / len(test_data)
        accuracies.append([mid_layer_size, accuracy])
        print("accuracy:", accuracy)
        with open(os.path.expanduser("~/brainside/pythonExperiment/hashed_one_layer_test.txt"), 'a+') as result_file:
            result = accuracies[-1]
            result_file.write(result[0].__str__() + " " + result[1].__str__() + "\n")


if __name__ == '__main__':
    main(30, 31, 30)
