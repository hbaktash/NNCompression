# Libraries
import errno
import math

# import networkx
import argparse
import pickle
import random
import numpy as np
import os

# from compressedNets import plot_handler
from compressedNets.file_handler import safe_open_w
from pythonExperiment import mnist_loader
from data import data_loader

import time

# constants
MIN_ACCURACY_FOR_DAMAGE = 0.95
EXTRA_EPOCHS_WITH_NO_DAMAGE = 50
MAX_DAMAGE_STEPS = 15
MIN_WAITING_FOR_UPDATE = 25
VERSION = "9"
REGULARIZATION = "L2"


def update_eta(step, initial_eta):
    return 2 * initial_eta / math.sqrt(step)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_prime_prime(z):
    return sigmoid(z) * (1 - sigmoid(z)) * (1 - 2 * sigmoid(z))


def sign(w):
    return np.sign(w)


def load_data(data_name):
    training_data = data_loader.load_data(data_name, "train")
    test_data = data_loader.load_data(data_name, "test")
    validation_data = None
    return training_data, test_data


class IterativePrunerNN(object):

    def __init__(self, sizes, edge_removal_ratio, data_name, max_damage_steps=MAX_DAMAGE_STEPS,
                 max_steps_to_update=MIN_WAITING_FOR_UPDATE,
                 extra_epochs_after_damages=EXTRA_EPOCHS_WITH_NO_DAMAGE,
                 version=VERSION,
                 regula=REGULARIZATION,
                 method_name="general_iterative",
                 save_mode=False):
        self.data_name = data_name
        self.num_layers = len(sizes)
        self.edge_removal_ratio = edge_removal_ratio
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        """ ***zero means damaged**** """
        self.damaged_weights = [np.ones((nex, pre)) for pre, nex in zip(sizes[:-1], sizes[1:])]
        self.update_damage_count = 0
        self.max_damage_steps = max_damage_steps
        self.max_steps_to_update = max_steps_to_update
        self.minimum_initial_wait = extra_epochs_after_damages
        self.extra_epochs_after_damages = extra_epochs_after_damages
        self.version = version
        self.regularization_method = regula
        self.method_name = method_name
        self.eta = 1.0
        self.lamda = 1.0
        self.instance = 1
        self.save_mode = save_mode

    # def get_weights_hist(self, epoch, update_count):
    #     all_in_one = [a[0] for a in self.get_all_in_one_arr(self.weights)]
    #     plot_handler.plot_hist(all_in_one, 1, "epoch "+str(epoch)+", pruning step"+str(update_count))

    def get_print_accuracies(self, training_data, test_data, validation_data, j):
        training_accuracy, test_accuracy, validation_accuracy = 0, 0, 0
        training_accuracy = self.evaluate(None, training_data) / len(training_data)
        if test_data:
            test_accuracy = (self.evaluate(test_data) / len(test_data))
        if validation_data:
            validation_accuracy = (self.evaluate(validation_data) / len(validation_data))
        print("\n "
              " ---------------------------------- "
              "\n")
        print("Epoch {0}: train   - {1}".format(j, training_accuracy))
        print("      test       -", test_accuracy)
        print("      validation -", validation_accuracy)
        return training_accuracy, test_accuracy, validation_accuracy

    def remaining_edges_ratio(self):
        non_zero_edges_ratio = sum([np.sum(w != 0) for w in self.weights]) / sum([w.size for w in self.weights])
        mask_non_zero_ratio = sum([np.sum(mask != 0) for mask in self.damaged_weights]) / sum(
            [mask.size for mask in self.damaged_weights])
        # if non_zero_edges_ratio != mask_non_zero_ratio:
        #     print("WTF", non_zero_edges_ratio, " ", mask_non_zero_ratio)
        return mask_non_zero_ratio

    def hessian_bp(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        # t1 = time.time()
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # t2 = time.time()
        # print("^^^   forward time", t2 - t1)

        # backward pass
        t1 = time.time()

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta_delta = self.cost_derivative(activations[-1], y) * sigmoid_prime_prime(zs[-1]) + sigmoid_prime(
            zs[-1]) ** 2
        # print(delta_delta.shape)
        # print(delta.shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_nabla_w[-1] = np.dot(delta_delta, activations[-2].transpose() ** 2)
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            spp = sigmoid_prime_prime(z)
            delta_delta = (sp ** 2) * np.dot((self.weights[-l + 1].transpose() ** 2), delta_delta) + spp * np.dot(
                self.weights[-l + 1].transpose(), delta)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_nabla_w[-l] = np.dot(delta_delta, activations[-l - 1].transpose() ** 2)
        # t2 = time.time()
        # print("^^^   backward time", t2 - t1)
        return nabla_nabla_w

    def get_hessians(self, data):
        hessians = [np.zeros((nex, pre)) for pre, nex in zip(self.sizes[:-1], self.sizes[1:])]
        for x, y in data:
            delta_delta_weights = self.hessian_bp(x, y)
            hessians = [current_hessian + delta_hessian for current_hessian, delta_hessian in
                        zip(hessians, delta_delta_weights)]
        hessians = [hessian_matrix / len(data) for hessian_matrix in hessians]
        return hessians

    def required_sparsity(self, pruning_step):
        return math.sqrt(math.sqrt(math.sqrt(math.sqrt(pruning_step / MAX_DAMAGE_STEPS))))

    def fraction_to_remove(self):
        fraction_to_remove = self.edge_removal_ratio * self.required_sparsity(self.update_damage_count)
        return fraction_to_remove

    def get_all_in_one_arr(self, list_off_ndarray):
        flattened_arrs = [ndarr.reshape((1, ndarr.size))[0] for ndarr in list_off_ndarray]
        flattened_arrs = [[(sal, layer, index) for sal, index in zip(flattened_sal, range(len(flattened_sal)))] for
                          flattened_sal, layer in zip(flattened_arrs, range(len(flattened_arrs)))]
        all_in_one = []
        for flat_arr in flattened_arrs:
            all_in_one = all_in_one + flat_arr
        return all_in_one

    def get_indices_for_removal(self, saliencies):
        # put the saliencies in an array
        flattened_sals = self.get_all_in_one_arr(saliencies)
        chunk_to_remove = math.floor(self.fraction_to_remove() * len(flattened_sals))
        print("chunk=", chunk_to_remove)
        sorted_sals_indexs = [(a[1], a[2]) for a in sorted(flattened_sals, key=lambda x: x[0])][:chunk_to_remove]
        # print(sorted_sals_indexs)
        remove_indices = [[a[1] for a in sorted_sals_indexs if a[0] == current_layer] for current_layer in
                          range(self.num_layers - 1)]
        return remove_indices

    def remove_low_saliencies(self, saliencies):
        """
        at step i, (i/n)^2 compression is achieved
        """
        fraction_to_remove = self.fraction_to_remove()
        print("fraction to remove : ", fraction_to_remove)
        indices = self.get_indices_for_removal(saliencies)
        for i, dmg_mat, index_list in zip(range(len(indices)), self.damaged_weights, indices):
            flat = dmg_mat.flatten()
            flat[index_list] = 0
            self.damaged_weights[i] = flat.reshape(dmg_mat.shape)

    def get_saliencies(self, training_data):
        pass

    def update_damaged_weights(self, training_data):
        # only updates if error is close to zero
        print("%%%%%% updating damages : {0}/{1}".format(self.update_damage_count, MAX_DAMAGE_STEPS))
        print("getting saliencies")
        saliencies = self.get_saliencies(training_data)
        print("*** damaged matrices updating ... ")
        print("before update :", [np.sum(dmgW) / dmgW.size for dmgW in self.damaged_weights])

        self.remove_low_saliencies(saliencies)

        print("after update  :", [np.sum(dmgW) / dmgW.size for dmgW in self.damaged_weights])
        print("total removal :",
              sum([np.sum(dmgW) for dmgW in self.damaged_weights]) / sum([dmgW.size for dmgW in self.damaged_weights]))

        """ here the damaged weights are set to zero """
        self.weights = [np.multiply(w, damage_matrix) for w, damage_matrix
                        in zip(self.weights, self.damaged_weights)]
        # print("actual zero weights  :", [np.sum(w != 0) / w.size for w in self.weights])
        print("saving damaged network in file...")
        # self.save_to_file("OBD_removal_step{}".format(self.update_damage_count))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def print_pre_run_info(self):
        print("**********RUNNING ", self.method_name, "with nhu", self.sizes[1],
              "\nremoval_ratio ", self.edge_removal_ratio,
              "\nregularization ", self.regularization_method, " lambda ", self.lamda,
              "\neta ", self.eta, "instance:", self.instance)

    def SGD(self, training_data, epochs=100, mini_batch_size=30, eta=1.0, lamda=1.0,
            min_accuracy_for_damage=0.95,
            test_data=None,
            validation_data=None,
            instance=1):
        n = len(training_data)
        epoch_accuracies = []
        initial_eta = eta
        self.eta = eta
        self.lamda = lamda
        self.instance = instance
        j = 0
        last_update = 0
        self.print_pre_run_info()
        while (j <= epochs) or self.update_damage_count <= MAX_DAMAGE_STEPS:
            j += 1
            # if j == 30:
            #     self.get_weights_hist(j, self.update_damage_count)
            # temp_count = 0
            # if self.update_damage_count == self.max_damage_steps:
            #     self.get_weights_hist(j, self.update_damage_count)
            #     temp_count = j
            # if j == temp_count + 30:
            #     self.get_weights_hist(j, self.update_damage_count)
            eta = update_eta(j, initial_eta)
            print("eta =", eta)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            """printing accuracies"""
            training_accuracy, test_accuracy, validation_accuracy = self.get_print_accuracies(training_data, test_data,
                                                                                              validation_data, j)
            epoch_accuracies.append(
                [training_accuracy, test_accuracy, validation_accuracy, self.remaining_edges_ratio()])
            # t0 = time.time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lamda, n)
            # t1 = time.time()
            # print("update of batches took -> ", t1 - t0)
            """ damage occurs when training gets to a good accuracy """
            if self.update_damage_count == self.max_damage_steps:
                print("PRUNING FINISHED\n--->FINAL phase of training... ({} extra epochs)".format(
                    self.extra_epochs_after_damages))
                epochs = j + self.extra_epochs_after_damages
                self.update_damage_count += 1
            elif self.update_damage_count < self.max_damage_steps:
                if (training_accuracy >= min_accuracy_for_damage) or (j - last_update >= self.max_steps_to_update):
                    if not (last_update == 0 and j < self.minimum_initial_wait):
                        last_update = j
                        self.update_damage_count += 1
                        self.save_to_file("iteration_{}_update_step_{}_pre".format(str(j), self.update_damage_count))
                        self.update_damaged_weights(training_data)
                        print("saving the network after update")
                        self.save_to_file("iteration_{}_update_step_{}_post".format(str(j), self.update_damage_count))
                        # print("updating took -> ", time.time() - t1)
                        # self.get_weights_hist()
                        # print("***accuracies right after update***")
                        # self.get_print_accuracies(training_data, test_data, validation_data, j)

        # print to file in the end
        mode = 'w+'
        with safe_open_w(os.path.expanduser(
                "~/brainside/data/results/epochs/version_{3}/{1}/epochs_{0}_regula_{5}_comp_{2}_inst{4}.txt".format(
                    self.method_name,
                    self.data_name,
                    self.edge_removal_ratio,
                    self.version,
                    instance,
                    self.regularization_method)),
                mode) as result_file:
            count = 0
            for epoch_accuracy in epoch_accuracies:
                result_file.write(
                    "\n-------------epoch #{0} comp to {4} \n train accuracy: {1}\n test accuracy  : {2}\ncompressed        : {3}".format(
                        count,
                        epoch_accuracy[0],
                        epoch_accuracy[1],
                        epoch_accuracy[3],
                        self.edge_removal_ratio))
                count += 1
        print("saving final network...")
        self.save_to_file("final_{}_net_instance_{}_with_{}_epochs".format(self.method_name, instance, str(j)))
        print(
            "Training has ended after {0} epochs.\ndata set: {3}\nfinal test accuracy  : {1}\nundamaged wights     : {2}".format(
                j + 1,
                self.evaluate(test_data) / len(test_data),
                sum([np.sum(w != 0) for w in self.weights]) / sum([w.size for w in self.weights]),
                self.data_name))

    def reg_term(self, w):
        if self.regularization_method == "L2":
            return w
        elif self.regularization_method == "L1":
            return sign(w)
        else:
            return np.zeros(w.shape)

    def update_mini_batch(self, mini_batch, eta, lamda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # t0 = time.time()
        t0 = time.time()
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        t1 = time.time()
        # print("\t nablas took -> ", t1 - t0)
        nabla_w = [np.multiply(nw, damage_matrix) for nw, damage_matrix in zip(nabla_w, self.damaged_weights)]
        self.weights = [w - (lamda * eta * self.reg_term(w)) / n - (eta / len(mini_batch)) * nw
                        for w, nw, damage_matrix in zip(self.weights, nabla_w, self.damaged_weights)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        # print("\t multiply took -> ", time.time() - t1)
        # print("batch time: ", time.time() - t0)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # t1 = time.time()
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # t2 = time.time()
        # print("^^^   forward time", t2 - t1)

        # backward pass
        t1 = time.time()
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        # t2 = time.time()
        # print("^^^   backward time", t2 - t1)
        return nabla_b, nabla_w

    def evaluate(self, test_data, training_data=None):
        if training_data:
            train_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                             for (x, y) in training_data]
            return sum(int(x == y) for (x, y) in train_results)
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def save_to_file(self, file_name):
        print("$$$$$$     saving ")
        path = os.path.expanduser(
            "~/brainside/data/results/saved_nets/version_{3}/{1}/{0}_regula_{4}_compression_{2}/".format(
                self.method_name,
                self.data_name,
                self.edge_removal_ratio,
                self.version,
                self.regularization_method))
        file_type = ".pkl"
        with safe_open_w(path + file_name + file_type, 'wb+') as net_file:
            pickle.dump(self, net_file)

    def save_final_result_to_file(self, training_data, test_data):
        final_accuracy = self.evaluate(test_data, None) / len(test_data)
        train_accu = self.evaluate(None, training_data) / len(training_data)
        with safe_open_w(os.path.expanduser(
                "~/brainside/data/results/final_results/version_{1}/{0}_{2}_compression_{3}_regula_{4}.txt".format(
                    self.method_name,
                    self.version,
                    self.data_name,
                    self.edge_removal_ratio,
                    self.regularization_method)),
                'a+') as f:
            f.write("instance {}:\ntest:{}\ntrain:{}\n".format(self.instance,
                                                               final_accuracy,
                                                               train_accu))


def get_parser(method_name):
    parser = argparse.ArgumentParser(description=method_name)
    parser.add_argument('--sizes', nargs='+', type=int, default=[100], metavar='hidden_units list',
                        help='hidden units list (default: 64)')
    parser.add_argument('--edge_removal_ratio', type=float, default=0.9, metavar="edge_removal_ratio"
                        , help='1 - compression')
    parser.add_argument('--data_name', type=str, default='mnist', metavar="data set name"
                        , help='dataset name')
    parser.add_argument('--dmg_steps', type=int, default=15, metavar="damage steps"
                        , help='convergance steps')
    parser.add_argument('--update_wait', type=int, default=25, metavar="waiting time for update"
                        , help='to prune after given steps')
    parser.add_argument('--extra_epochs', type=int, default=50, metavar="epochs after last update"
                        , help='train steps after sparsifying')
    parser.add_argument('--version', type=str, default='X', metavar='code version'
                        , help='code version for saving to file')
    parser.add_argument('--regula', type=str, default='L2', metavar='regula method'
                        , help='L1 or L2 or smth else for no regula')
    parser.add_argument('--method_name', type=str, default=method_name, metavar='method name'
                        , help='name of the method')

    parser.add_argument('--batch_size', type=int, default=100, metavar='batch size'
                        , help='batch size')
    parser.add_argument('--eta', type=float, default=1.0, metavar='learning rate'
                        , help='learning rate')
    parser.add_argument('--lamda', type=float, default=1.0, metavar='regula rate'
                        , help='regula rate')
    parser.add_argument('--instance', type=int, default=1, metavar='instance'
                        , help='traning number')
    parser.add_argument('--min_accuracy_for_update', type=float, default=0.96, metavar='min accuracy needed'
                        , help='min accuracy for update')
    return parser

