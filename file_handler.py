import errno
import math
import numpy as np
import os
import pickle


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open_w(path, mode):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, mode)


def generate_args_file(method_name: str, instances: int, version: str, regula):
    path = os.path.expanduser("~/brainside/compressedNets/arguments_{}_{}.txt".format(method_name, version))
    sizes = [30]
    sizes_string = ""
    for size in sizes:
        sizes_string = sizes_string + str(size) + " "
    compressions = [0.0, 0.5]  #  [0.993, 0.995, 0.997, 0.998] # [0.9, 0.95, 0.97, 0.98, 0.985, 0.988, 0.989, 0.99]  # [0.99, 0.993, 0.995, 0.997, 0.998]
    data_names = ["mnist", "mnist_background_images", "mnist_background_random", "mnist_rotation_normalized"]
    damage_steps = 15
    update_wait = 25
    extra_epochs = 50
    # version
    # regula
    # method_name
    batch_size = 100
    eta = 3.0
    lamda = 1.0
    min_accuracy_for_update = 0.99
    argument_lines = []
    for regula in ["L2"]:
        for instance in range(instances):
            for removal_ratio in compressions:
                for data_name in data_names:
                    arg_string = "--sizes " + sizes_string + \
                                 "--edge_removal_ratio " + str(removal_ratio) + \
                                 " --data_name " + data_name + \
                                 " --dmg_steps " + str(damage_steps) + \
                                 " --update_wait " + str(update_wait) + \
                                 " --extra_epochs " + str(extra_epochs) + \
                                 " --version " + version + \
                                 " --regula " + regula + \
                                 " --method_name " + method_name + \
                                 " --batch_size " + str(batch_size) + \
                                 " --eta " + str(eta) + \
                                 " --lamda " + str(lamda) + \
                                 " --instance " + str(instance) + " --min_accuracy_for_update " + str(
                        min_accuracy_for_update)
                    argument_lines.append(arg_string + "\n")
    with safe_open_w(path, 'w+') as arg_file:
        for arg_line in argument_lines:
            arg_file.write(arg_line)


def load_network_from_results(method_name, regula, compression, data_name, version, instance):
    path = os.path.expanduser(
        "~/brainside/data/results/saved_nets/version_{0}/{1}/{2}_regula_{3}_compression_{4}/".format(version, data_name,
                                                                                                     method_name,
                                                                                                     regula,
                                                                                                     str(compression)))
    file_name = "final_{0}_net_instance_{1}.pkl".format(method_name, str(instance))
    with open(path + file_name, 'rb') as f:
        return pickle.load(f)


def load_for_hist(j, update, pre='post'):
    path = os.path.expanduser("~/brainside/data/results/saved_nets/for_histogram/iteration_{}_update_step_{}_{}.pkl".format(j,update, pre))
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_for_hist2(j, update, regula: str, pre='post'):
    path = os.path.expanduser("~/brainside/data/results/saved_nets/version_for_histogram/mnist/mag_prune_regula_{}_compression_0.99/iteration_{}_update_step_{}_{}.pkl".format(regula, j,update, pre))
    with open(path, 'rb') as f:
        return pickle.load(f)


def fetch_answers(method_name: str, instances: int, version: str, regula):
    compressions = [0.0, 0.5]  # [0.9, 0.95, 0.97, 0.98, 0.985, 0.988, 0.989, 0.99]  # [0.99, 0.993, 0.995, 0.997, 0.998]  #
    data_names = ["mnist", "mnist_background_images", "mnist_background_random", "mnist_rotation_normalized"]
    final_result_files = []
    for data_name in data_names:
        for compression in compressions:
            file_name = os.path.expanduser(
                "~/brainside/data/results/final_results/version_{0}/{1}_{2}_compression_{3}_regula_{4}.txt".format(
                    version, method_name, data_name, compression, regula))
            try:
                with open(file_name, 'r') as f:
                    lines = f.readlines()
                    test_accus = []
                    for line in lines:
                        if len(line.split(" ")) == 2:
                            pass
                        elif line.split(":")[0] == "test":
                            test_accus.append(math.floor(10000 * float(line.split(":")[1])) / 100)
                    print(method_name + "_" + data_name + "_compression_" + str(compression) + " " + regula, ":\n",
                          test_accus)
                    print("average:\n", math.floor(100 * np.mean(test_accus)) / 100)
            except FileNotFoundError:
                print(
                    method_name + "_" + data_name + "_compression_" + str(compression) + " " + regula + " was not made")


def get_statistics(input_str: str):
    lines = input_str.split("\n")
    for line in lines:
        line = line.replace("]", "")
        line = line.replace("[", "")
        accus = list(map(float, line.split(",")))
        std = stdv(accus)
        mean = np.mean(accus)
        print(math.floor(std*100)/100)


def stdv(data):
    data = np.array(data)
    std = math.sqrt(np.sum((data - np.mean(data))**2)/(len(data)-1))
    return std

