import os
import sys

import argparse

from compressedNets import iterative_pruing_NN
import numpy as np
from data import data_loader
np.histogram_bin_edges

class BD_Net(iterative_pruing_NN.IterativePrunerNN):
    def __init__(self, sizes,
                 edge_removal_ratio,
                 data_name,
                 max_damage_steps,
                 max_steps_to_update,
                 extra_epochs_after_damages,
                 version,
                 regula,
                 method_name):
        super().__init__(sizes, edge_removal_ratio, data_name, max_damage_steps, max_steps_to_update,
                         extra_epochs_after_damages, version, regula, method_name)

    def get_saliencies(self, train_data):
        hessians = self.get_hessians(train_data)
        print("saliencies ...")
        saliencies = [np.multiply(w ** 2, np.abs(hessian)) for w, hessian in zip(self.weights, hessians)]
        return saliencies


if __name__ == '__main__':
    parser = iterative_pruing_NN.get_parser("OBD")
    args = parser.parse_args()
    print(args)
    net = BD_Net(sizes=[28*28]+args.sizes+[10],
                 edge_removal_ratio=args.edge_removal_ratio,
                 data_name=args.data_name,
                 max_damage_steps=args.dmg_steps,
                 max_steps_to_update=args.update_wait,
                 extra_epochs_after_damages=args.extra_epochs,
                 version=args.version,
                 regula=args.regula,
                 method_name=args.method_name)
    training_data, test_data = iterative_pruing_NN.load_data(args.data_name)
    net.SGD(training_data,
            epochs=300,
            mini_batch_size=args.batch_size,
            eta=args.eta,
            lamda=args.lamda,
            min_accuracy_for_damage=args.min_accuracy_for_update,
            test_data=test_data,
            validation_data=None,
            instance=args.instance)
    net.save_final_result_to_file(training_data, test_data)
