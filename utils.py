import random
import torch
import numpy as np
import queue
import torch.nn as nn
import os
import copy
import torchvision.transforms as transforms
from torchvision import datasets
from CIFAR10_models.vgg import vgg16_bn
from CIFAR10_models.resnet import resnet18
from CIFAR10_models.googlenet import GoogLeNet
from torchvision.models import resnet50, densenet121
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools


EPSILON = 8/256

def setup_devices():
    """
    Set up the available CUDA devices for parallel processing.

    Returns:
        list: A list of torch.device objects representing available CUDA devices.
    """
    devices = []
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        print(f'Using {num_gpus} GPUs')
        for i in range(num_gpus):
            devices.append(torch.device(f'cuda:{i}'))
    return devices


def generate_real_value(cond_type, img_dim):
    """
    Generate a real value based on the given condition type and image dimension.

    Args:
        cond_type (str): A string representing the condition type.
        img_dim (int): An integer representing the image dimension. This is only used when
            the cond_type is "CENTER".

    Returns:
        float: A real value generated based on the given condition type and image dimension.
    """
    if cond_type == "SCORE_DIFF":
        real_value = random.uniform(-0.02, 0.3)
    else:
        real_value = random.random()
    return real_value


def generate_center_matrix(img_dim):
    """
    Generate a center matrix of the specified image dimension.

    This function creates a square matrix of size img_dim x img_dim, where each element
    represents the l_inf distance of the element from the center of the matrix.

    Args:
        img_dim (int): An integer representing the image dimension.

    Returns:
        torch.Tensor: A torch.Tensor representing the center matrix with shape (img_dim, img_dim).
    """
    center_matrix = torch.zeros((img_dim, img_dim), device='cpu')
    interval = [0, img_dim - 1]
    distance = 0

    for m in range(img_dim // 2):
        for i in range(img_dim):
            for j in range(img_dim):
                if i in interval or j in interval:
                    if center_matrix[i, j] == 0:
                        center_matrix[i, j] = abs(distance + 1 - (img_dim / 2))
        interval = [interval[0] + 1, interval[1] - 1]
        distance += 1

    return center_matrix


def generate_random_condition(img_dim):
    """
    Generate a random condition with a condition type, comparison operator, and real value.

    Args:
        img_dim (int): An integer representing the image dimension. This is only used when
            the selected condition type is "CENTER".

    Returns:
        list: A list containing the randomly generated condition type (str), comparison operator (str),
            and real value (float).
    """
    cond_type = random.choice(["MIN", "MAX", "MEAN", "SCORE_DIFF"])
    comparison_operator = random.choice([">", "<"])

    if cond_type == "SCORE_DIFF":
        real_value = random.uniform(-0.02, 0.3)
    else:
        real_value = random.random()

    return [cond_type, comparison_operator, real_value]


def argsort(seq):
    """
    Return a list of indices that would sort the input sequence in ascending order.

    Args:
        seq (list or numpy.array): A sequence of numeric values to be sorted.

    Returns:
        list: A list of indices that would sort the input sequence in ascending order.
    """
    return sorted(range(len(seq)), key=lambda i: seq[i])


# def create_loc_pert_dict(img_x, mid_values):
#     """
#     Create a dictionary with pixel locations as keys and a list of perturbation types as values.
#
#     Args:
#         img_x (torch.Tensor): The input image tensor.
#         mid_values (list): A list of mid values for each color channel (e.g., [0.5, 0.5, 0.5]).
#
#     Returns:
#         dict: A dictionary with pixel locations as keys and a list of perturbation types as values.
#     """
#     pixel_pert_dict = {}
#     img_shape = img_x.shape[-1]
#
#     for x in range(img_shape):
#         for y in range(img_shape):
#             pert_type_list = []
#             diff_list = []
#
#             for c in range(3):
#                 channel_value = img_x[0, c, x, y].item()
#                 diff_list.append(abs(channel_value - mid_values[c]))
#                 pert_type_list.append("MAX" if channel_value < mid_values[c] else "MIN")
#
#             sorted_diff_list = argsort(diff_list)
#             all_pert_list = [tuple(pert_type_list)]
#
#             for idx in sorted_diff_list:
#                 new_pert = copy.deepcopy(pert_type_list)
#                 new_pert[idx] = "MIN" if pert_type_list[idx] == "MAX" else "MAX"
#                 all_pert_list.append(tuple(new_pert))
#
#             opposite_pert_type_list = ["MIN" if elem == "MAX" else "MAX" for elem in pert_type_list]
#             sorted_diff_list.reverse()
#
#             for idx in sorted_diff_list:
#                 new_pert = copy.deepcopy(opposite_pert_type_list)
#                 new_pert[idx] = "MIN" if opposite_pert_type_list[idx] == "MAX" else "MAX"
#                 all_pert_list.append(tuple(new_pert))
#
#             all_pert_list.append(tuple(opposite_pert_type_list))
#             pixel_pert_dict[(x, y)] = all_pert_list
#
#     return pixel_pert_dict


def create_sorted_pert_list(amount_square):
    """
    Create a sorted list of pixel perturbation types based on the difference of the perturbation type
    from the original pixel as the primary key and the distance from the center as a secondary key.

    Args:
        lmh_dict (dict): A dictionary containing the 'min_values', 'mid_values', and 'max_values' for the perturbations.
        - FOR NOW WE REMOVE IT

    Returns:
        #list: A sorted list of tuples containing pixel locations and perturbation types.

    """
    # img_shape = img_x.shape[-1]
    # distance_list = [i for i in range(0, img_shape) for _ in range(2)]

    # possible_loc_pert_list = sorted(
    #     {(i, k) for i in distance_list for k in distance_list},
    #     key=lambda x: abs((img_shape / 2) - x[0]) + abs((img_shape / 2) - x[1])
    # )

    # loc_pert_dict = create_loc_pert_dict(img_x, lmh_dict['mid_values'])
    possible_pert = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    possible_pert_list_tuples = list(itertools.product(possible_pert, repeat=int(amount_square)))
    possible_pert_list = [list(ele) for ele in possible_pert_list_tuples]

    return possible_pert_list


def get_orig_confidence(model, img_x, img_y, device):
    """
    Compute the original confidence of the model's prediction for a given image.

    Args:
        model (nn.Module): The neural network model to evaluate the image.
        img_x (torch.Tensor): The input image tensor.
        img_y (torch.Tensor): The ground truth label tensor for the image.
        device (torch.device): The device to run the model on, e.g., 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: The original confidence of the model's prediction for the input image.
    """
    softmax = nn.Softmax(dim=1)
    predictions_vector = softmax(model(img_x).data)
    orig_confidence = predictions_vector[0][img_y.item()].to(device)
    return orig_confidence


def create_pert_type_to_idx_dict():
    """
    Create a dictionary to map perturbation types to their corresponding indices.

    The perturbation types are represented as tuples with three elements, each being
    either 'MAX' or 'MIN', representing the maximum or minimum value for each of the
    three color channels (R, G, B).

    Returns:
        dict: A dictionary mapping perturbation types to their corresponding indices.
    """
    return {
        (0, 0, 0): 0,
        (1, 0, 0): 1,
        (0, 1, 0): 2,
        (0, 0, 1): 3,
        (1, 1, 0): 4,
        (0, 1, 1): 5,
        (1, 0, 1): 6,
        (1, 1, 1): 7,
    }


def create_neighbors_list():
    return (
        [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
         [(0, 0, 0), (1, 1, 0), (1, 0, 1)],
         [(1, 1, 0), (0, 0, 0), (0, 1, 1)],
         [(1, 0, 1), (0, 1, 1), (0, 0, 0)],
         [(0, 1, 0), (1, 0, 0), (1, 1, 1)],
         [(1, 1, 1), (0, 0, 1), (0, 1, 0)],
         [(0, 0, 1), (1, 1, 1), (1, 0, 0)],
         [(0, 1, 1), (1, 0, 1), (1, 1, 0)]]
    )


# new function for create square perturbation after replacing one square with his neighbor
def create_new_neighbors_pert(neighbor, num_square, curr_pert):
    curr_pert[num_square] = neighbor
    return curr_pert


# change this for create only perturbation and current confidence queue
def initialize_pixels_conf_queues(pert_img, curr_confidence):
    """
    Initialize two queue objects with initial pixel location, perturbation type and current confidence.

    This function creates two queue.Queue objects, both initialized with a tuple containing
    a pixel location (x, y), the perturbation type, and the current confidence. The queues are
    returned in the same order they were created.

    Parameters:
    x (int): The x-coordinate of the pixel location.
    y (int): The y-coordinate of the pixel location.
    pert_type (list[str]): The type of perturbation to be applied.
    curr_confidence (float): The current confidence score.

    Returns:
    tuple: A tuple containing two queue.Queue objects, each initialized with a tuple
           containing the pixel location, perturbation type and current confidence.
    """

    pert_queue = queue.Queue()
    pert_queue.put((pert_img, curr_confidence))

    return pert_queue


def is_correct_prediction(model, img_x, img_y):
    """
    Check if the model's prediction for the given image is correct.

    Args:
        model (torch.nn.Module): The neural network model to use for prediction.
        img_x (torch.Tensor): The input image tensor.
        img_y (torch.Tensor): The true label of the input image.

    Returns:
        bool: True if the model's prediction is correct, False otherwise.
    """
    softmax = nn.Softmax(dim=1)
    predictions_vector = softmax(model(img_x).data)
    pred = torch.argmax(predictions_vector)
    return pred.item() == img_y.item()


def update_min_confidence_dict(min_confidence_dict, permutation, curr_confidence):
    """
    Update the minimum confidence dictionary with the current confidence
    for a given pixel location (x, y).

    Args:
        min_confidence_dict (dict): A dictionary containing the minimum confidences for pixel locations.
        x (int): The x-coordinate of the pixel location.
        y (int): The y-coordinate of the pixel location.
        curr_confidence (float): The current confidence value for the pixel location.
    """
    if permutation in min_confidence_dict:
        min_confidence_dict[permutation] = min(curr_confidence, min_confidence_dict[permutation])
    else:
        min_confidence_dict[permutation] = curr_confidence


def get_rgb(row, col, img_x):
    img_shape = img_x.shape[-1]
    size_square = int(img_shape / 2)
    rgb = []
    for c in range(3):
        rgb.append(torch.mean(img_x[0, c, size_square * row:(size_square * row + size_square - 1),
                              size_square * col: size_square * col + size_square - 1]))
    return rgb


def check_cond(cond, img_x, orig_confidence, confidence, amount_square):
    """
    Check if a condition is satisfied for a pixel in the input image.

    Args:
        cond (tuple): A tuple containing the condition to be checked.
        img_x (torch.Tensor): The input image tensor.
        orig_confidence (torch.Tensor): The original confidence of the true class.
        confidence (torch.Tensor): The confidence of the true class after perturbation.
        center_matrix (torch.Tensor): A matrix representing the distance of each pixel to the image center.

    Returns:
        bool: True if the condition is satisfied, False otherwise.
    """

    def RGB_per_squre(row, col, img_x):
        R, G, B = get_rgb(row, col, img_x)
        return list((min(R, G, B), max(R, G, B), ((R + G + B) / 3)))

    num_row = 3 if amount_square > 8 else 2
    matrix_rgb = [[[] for _ in range(int(amount_square / num_row))] for _ in range(int(num_row))]

    for row in range(num_row):
        for col in range(int(amount_square / num_row)):
            matrix_rgb[row][col] = RGB_per_squre(row, col, img_x)

    confidence_diff = (orig_confidence - confidence).item()
    condition_type, comparison_operator, value = cond

    def bigger_than(cond_type, value):
        counter = 0
        for row in range(num_row):
            for col in range(int(amount_square / num_row)):
                if (cond_type == "MIN") and (matrix_rgb[row][col][0] > value):
                    counter += 1
                elif (cond_type == "MAX") and (matrix_rgb[row][col][1] > value):
                    counter += 1
                elif (cond_type == "MEAN") and (matrix_rgb[row][col][2] > value):
                    counter += 1

        if counter >=  (amount_square - 1):
            return True

    def smaller_than(cond_type, value):
        counter = 0
        for row in range(num_row):
            for col in range(int(amount_square / num_row)):
                if (cond_type == "MIN") and (matrix_rgb[row][col][0] < value):
                    counter += 1
                elif (cond_type == "MAX") and (matrix_rgb[row][col][1] < value):
                    counter += 1
                elif (cond_type == "MEAN") and (matrix_rgb[row][col][2] < value):
                    counter += 1

        if counter >= (amount_square - 1):
            return True

    if condition_type == "MIN" or condition_type == "MAX" or condition_type == "MEAN":
        return bigger_than(condition_type, value) if comparison_operator == ">" else smaller_than(condition_type, value)
    elif condition_type == "SCORE_DIFF":
        return confidence_diff > value if comparison_operator == ">" else confidence_diff < value


def get_intarvel(row, col, img_shape, amount_square):
    interval_x_start = row * int(img_shape / 2)
    interval_x_end = row * int(img_shape / 2) + int(img_shape / 2)
    interval_y_start = col * int(img_shape / (amount_square / 2))
    interval_y_end = col * int(img_shape / (amount_square / 2)) + int(img_shape / (amount_square / 2))
    if (col == ((amount_square / 2) - 1)) and (interval_y_end != img_shape):
        interval_y_end = img_shape
    return [interval_x_start, interval_x_end, interval_y_start, interval_y_end]


def try_perturb_img(model, img_x, img_y, perturbation, device, amount_square):
    """
    Try perturbing a pixel using the specified perturbation type and evaluate the impact on the model's prediction.

    Args:
        x (int): The x-coordinate of the pixel to perturb.
        y (int): The y-coordinate of the pixel to perturb.
        model (nn.Module): The trained model to evaluate the perturbation on.
        img_x (torch.Tensor): The input image tensor.
        img_y (torch.Tensor): The ground truth label tensor for the input image.
        pert_type (list): A list of strings representing the perturbation type for each color channel (e.g., ['MIN', 'MAX', 'MIN']).
        lmh_dict (dict): A dictionary containing the 'min_values' and 'max_values' used for the perturbation.
        device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        bool: True if the perturbation causes a misclassification, False otherwise.
        int: The number of queries performed during the perturbation.
        torch.Tensor: The confidence of the true class after perturbation.
    """
    n_queries_pert = 0
    pert_img = torch.clone(img_x)
    img_shape = img_x.shape[-1]
    # perturbation is tuple perturbation for amount_squares ((,,),(,,),(,,),(,,))
    # pert = perturbation()
    num_row = 3 if amount_square > 8 else 2
    for row in range(num_row):
        for col in range(int(amount_square/num_row)):
            output = get_intarvel(row, col, img_shape, amount_square)
            for c, pert in enumerate(perturbation[num_row * row + 1 * col]):  # (0,1,0)
                if pert == 0:
                    pert_img[0, c, output[0]:output[1], output[2]:output[3]] = \
                        pert_img[0, c, output[0]:output[1], output[2]:output[3]] - EPSILON

                else:
                    pert_img[0, c, output[0]:output[1], output[2]:output[3]] = \
                        pert_img[0, c, output[0]:output[1], output[2]:output[3]] + EPSILON
    pert_img[pert_img < 0] = 0
    pert_img[pert_img > 1] = 1
    # for c, pert in enumerate(pert_type):
    #     if pert == "MIN":
    #         pert_img[0, c, x, y] = lmh_dict['min_values'][c]
    #     else:
    #         pert_img[0, c, x, y] = lmh_dict['max_values'][c]
    n_queries_pert += 1
    softmax = nn.Softmax(dim=1)
    predictions_vector = softmax(model(pert_img).data)
    pred = torch.argmax(predictions_vector)
    confidence = predictions_vector[0][img_y.item()].to(device)
    if pred.item() != img_y.item():
        return True, n_queries_pert, confidence

    return False, n_queries_pert, confidence


def create_low_mid_high_values_dict(mean, std):
    """
    Generate a dictionary containing normalized 'max', 'mid', and 'min' values based on given mean and standard deviation.

    The function \s a dictionary with keys 'max_values', 'mid_values', and 'min_values'. Each key corresponds to
    a numpy array that results from the normalization operation (value - mean) / std.

    The 'max_values' key corresponds to maximal value for each channel after normalization.
    The 'mid_values' key corresponds to middle value for each channel after normalization.
    The 'min_values' key corresponds to minimal value for each channel after normalization.

    Args:
        mean (numpy.array): A numpy array representing the mean values.
        std (numpy.array): A numpy array representing the standard deviations.

    Returns:
        dict: A dictionary with keys 'max_values', 'mid_values', and 'min_values'. Each key corresponds to a numpy
        array of normalized values based on the input mean and standard deviation.
    """
    low_mid_high_values_dict = {}
    low_mid_high_values_dict["max_values"] = (np.array([1, 1, 1]) - np.array(mean)) / np.array(std)
    low_mid_high_values_dict["mid_values"] = (np.array([0.5, 0.5, 0.5]) - np.array(mean)) / np.array(std)
    low_mid_high_values_dict["min_values"] = (np.array([0, 0, 0]) - np.array(mean)) / np.array(std)
    return low_mid_high_values_dict


# def try_perturb_pixel_finer_granularity(x, y, model, img_x, img_y, g, mean_norm, std_norm, device):
#     """
#     Try perturbing a pixel with finer granularity and evaluate the impact on the model's prediction.
#
#     Args:
#         x (int): The x-coordinate of the pixel to perturb.
#         y (int): The y-coordinate of the pixel to perturb.
#         model (nn.Module): The trained model to evaluate the perturbation on.
#         img_x (torch.Tensor): The input image tensor.
#         img_y (torch.Tensor): The ground truth label tensor for the input image.
#         g (int): The granularity level for generating finer perturbations.
#         mean_norm (list[float]):  The mean values for each channel used in image normalization.
#         std_norm (list[float]):  The standard deviation values for each channel used in image normalization.
#         device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').
#
#     Returns:
#         bool: True if the perturbation causes a misclassification, False otherwise.
#         int: The number of queries performed during the perturbation.
#         torch.Tensor: The confidence of the true class after perturbation.
#     """
#     n_queries_pert = 0
#     pert_img = torch.clone(img_x)
#     finer_pert_granularity_list = generate_finer_granularity(g)
#     norm_finer_pert_granularity_list = [[(val - mean_norm[i]) / std_norm[i] for i, val in enumerate(row)] \
#                                         for row in finer_pert_granularity_list]
#
#     softmax = nn.Softmax(dim=1)
#
#     for pert in norm_finer_pert_granularity_list:
#         pert_img[0, :, x, y] = torch.tensor(pert)  # Apply the perturbation to all channels at once
#         n_queries_pert += 1
#         predictions_vector = softmax(model(pert_img).data)
#         pred = torch.argmax(predictions_vector)
#         confidence = predictions_vector[0][img_y.item()].to(device)
#
#         if pred.item() != img_y.item():
#             return True, n_queries_pert, confidence
#
#     return False, n_queries_pert, confidence


# def generate_finer_granularity(g):
#     """
#     Generate a list of color perturbations with finer granularity.
#
#     Args:
#         g (int): The granularity level. Higher values produce a more fine-grained list of perturbations.
#
#     Returns:
#         list: A list of color perturbations with finer granularity.
#     """
#     finer_granularity_list = []
#     n_steps = 2 ** g
#
#     for i in range(0, n_steps + 1):
#         for j in range(0, n_steps + 1):
#             for k in range(0, n_steps + 1):
#                 r, g, b = i / n_steps, j / n_steps, k / n_steps
#
#                 if r not in [0, 1] or g not in [0, 1] or b not in [0, 1]:
#                     finer_granularity_list.append([r, g, b])
#
#     return finer_granularity_list


def get_data_set(args, is_train=True):
    """
    Loads the specified dataset and applies the necessary transformations.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.
            - args.data_set (str): The name of the dataset to load. Supported options are "cifar10" and "imagenet".
            - args.imagenet_dir (str, optional): The directory containing the ImageNet dataset images, if applicable.
            - args.mean_norm (list of float, optional): The list of mean values for each channel for normalization.
            - args.std_norm (list of float, optional): The list of standard deviation values for each channel for normalization.
        is_train (bool, optional): If True, the function loads the training dataset. Otherwise, it loads the test dataset.
            Default is True.

    Returns:
        dataset (torch.utils.data.Dataset): The loaded and pre-processed dataset.
        img_dim (int): The dimensions (width and height) of the images in the dataset.

    Raises:
        Exception: If the ImageNet dataset is selected but the 'imagenet_dir' argument is not provided or the directory does not exist.
    """
    if args.data_set == "cifar10":
        img_dim = 32
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=args.mean_norm, std=args.std_norm)])
        train_data = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
        print('after download cifar10')
    elif args.data_set == "imagenet":
        img_dim = 224
        if args.imagenet_dir is None:
            raise Exception("imagenet_dir must be not None")
        if not os.path.exists('./' + args.imagenet_dir):
            raise Exception("can't find the directory for ImageNet")
        train_data = datasets.ImageFolder(
            os.path.join(args.imagenet_dir),
            transforms.Compose([
                transforms.Resize(img_dim),
                transforms.CenterCrop(img_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
            ]))
    return train_data, img_dim


def load_model(model_name):
    """
    Loads a pre-trained model given its name and transfers it to the specified device.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (torch.nn.Module): The loaded and pre-trained model, and set to evaluation mode.
    """
    # CIFAR-10
    if model_name == "vgg16":
        model = vgg16_bn()
        model.load_state_dict(torch.load("CIFAR10_models/vgg16_bn.pt", map_location='cpu'))
    elif model_name == "resnet18":
        model = resnet18()
        model.load_state_dict(torch.load("CIFAR10_models/resnet18.pt", map_location='cpu'))
    elif model_name == "GoogLeNet":
        model = GoogLeNet()
        model.load_state_dict(torch.load("CIFAR10_models/googlenet.pt", map_location='cpu'))

    # ImageNet
    elif model_name == "resnet50":
        model = resnet50(pretrained=True)
    elif model_name == "densenet121":
        model = densenet121(pretrained=True)

    model.eval()
    return model


def update_best_program(queue_proc, best_program, best_queries):
    """
    Updates the best program and its corresponding query count based on the results from the queue.

    Args:
        queue_proc (Queue): A queue containing tuples of (program, queries) from the multiprocessing results.
        best_program (Program): The current best program.
        best_queries (int): The current best query average.

    Returns:
        best_program (Program): The updated best program.
        best_queries (int): The updated best query average.
    """
    while not queue_proc.empty():
        program, queries = queue_proc.get()
        if queries < best_queries:
            best_queries = queries
            best_program = program
    return best_program, best_queries


def write_program_results(args, class_idx, best_program, best_queries):
    """
    Write the results of the program synthesis for a given class to a text file.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        class_idx (int): Index of the class for which the program has been synthesized.
        best_program (Program): Best synthesized program for the class.
        best_queries (float): Average number of queries on the training set for the best program.
    """
    with open(f"{args.model}_{args.data_set}.txt", 'a+') as f:
        f.write(f"class: {class_idx}\n")
        f.write(f"cond 1: {best_program.cond_1}\n")
        f.write(f"cond 2: {best_program.cond_2}\n")
        # f.write(f"cond 3: {best_program.cond_3}\n")
        # f.write(f"cond 4: {best_program.cond_4}\n")
        f.write(f"average number of queries on training set: {best_queries}\n\n")


def select_n_images(num_synthesis_images, true_label, data_loader, model, max_g, g, lmh_dict, mean_norm, std_norm,
                    device, amount_square):
    """
    Selects n images from a data loader such that a successful L-infinite attack can be performed on the selected images.

    Args:
        num_synthesis_images (int): The number of images to select.
        true_label (int): The true label to be matched.
        data_loader (DataLoader): A PyTorch DataLoader object containing the image data.
        model (nn.Module): A PyTorch model used for predictions.
        max_g (int): The maximum number of pixels to perturb with finer granularity.
        g (int): The level of granularity.
        lmh_dict (dict): A dictionary containing the 'min_values', 'mid_values', and 'max_values' for the perturbations.
        mean_norm (list[float]):  The mean values for each channel used in image normalization.
        std_norm (list[float]):  The standard deviation values for each channel used in image normalization.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        list: A list of successful image indices.
    """
    model.to(device)
    successful_indices = []

    with tqdm(total=num_synthesis_images, desc="Creating data set for synthesis",
              bar_format="{l_bar}{bar:10}{r_bar}") as progress_bar:
        for batch_idx, (data, target) in enumerate(data_loader):
            print("try image")
            is_success = False
            img_x, img_y = data.to(device), target.to(device)
            if img_y.item() != true_label:
                continue
            if not is_correct_prediction(model, img_x, img_y):
                continue
            
            # got here if we found correctly classified image 
            possible_perturbations = create_sorted_pert_list(amount_square)  # create a basic list L with all possible
            # perturbations, each item is ((),(),(),())
            possible_perturbations.append("STOP")
            # min_confidence_dict = {}

            for pert_img in possible_perturbations:  # perturbation is a ((),(),(),())
                if is_success:
                    break

                if pert_img == "STOP":
                    continue
                is_success, queries, curr_confidence = try_perturb_img(model, img_x, img_y, pert_img, device, amount_square)


            if is_success:
                successful_indices.append(batch_idx)
                progress_bar.update(1)

                if len(successful_indices) == num_synthesis_images:
                    return successful_indices


def update_results_df(results_df, results_path, batch_idx, class_idx, is_success, n_queries, pert_img):
    """
    Update the results DataFrame with the current batch's success status and queries, and save it to a CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing results for each batch.
        results_path (str): Path to the directory where the results CSV file should be saved.
        batch_idx (int): Index of the current batch.
        class_idx (int): Index of the current class.
        is_success (bool): Whether the current batch was successful or not.
        n_queries (int): Number of queries for the current batch.
        pert_img(int): tuple of the perturbation.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    result_row = {
        "batch_idx": batch_idx,
        "class": class_idx,
        "is_success": is_success,
        "queries": n_queries if is_success else -1,
        "pert_img": pert_img if is_success else -1
    }
    results_df = results_df._append(result_row, ignore_index=True)
    results_df.to_csv(f"{results_path[2:]}/class_{class_idx}.csv")

    return results_df

# def try_perturb_few_pixels(max_k, few_pixel_list, model, img_x, img_y, curr_n_queries, max_queries, lmh_dict, device):
#     """
#     Attempts to perturb a few pixels in the image to cause misclassification.
#
#     Parameters:
#     max_k (int): Maximum number of pixels that can be perturbed.
#     few_pixel_list (list): List of pixels that can be perturbed. Each entry in the list is a tuple with pixel location and perturbation type.
#     model (torch.nn.Module): The PyTorch model to use for prediction.
#     img_x (torch.Tensor): The input image tensor to be perturbed.
#     img_y (torch.Tensor): The true label of the image.
#     curr_n_queries (int): The current number of queries made to the model.
#     max_queries (int): The maximum number of queries allowed.
#     lmh_dict (dict): Dictionary holding the min and max values for pixel perturbation.
#     device (str or torch.device): The device (CPU or GPU) where the computations will be performed.
#
#     Returns:
#     tuple: A tuple containing:
#         - bool: True if the perturbation caused misclassification, False otherwise.
#         - int: The total number of queries made to the model.
#         - float: The confidence of the model's prediction for the true label.
#         - int: The number of pixels perturbed.
#     """
#     n_possible_pert = min(len(few_pixel_list), 100)
#     weights_pert = [(2 * n_possible_pert - 2 * i + 1) / n_possible_pert ** 2 for i in range(1, n_possible_pert + 1)]
#     n_queries_pert = 0
#     curr_k = 2
#     n_queries_k = 0
#     while curr_n_queries + n_queries_pert < max_queries:
#         n_queries_k += 1
#         if n_queries_k > 1000 and curr_k < max_k:
#             curr_k = min(curr_k + 1, max_k)
#             n_queries_k = 1
#
#         pert_img = torch.clone(img_x)
#         n_queries_pert += 1
#         sampled_pert = random.choices(few_pixel_list[:n_possible_pert], weights=weights_pert, k=curr_k)
#         for loc_pert in sampled_pert:
#             x, y = loc_pert[0]
#             pert_type = loc_pert[1]
#             for c in range(3):
#                 if pert_type[c] == "MIN":
#                     pert_img[0][c][x][y] = lmh_dict['min_values'][c]
#                 else:
#                     pert_img[0][c][x][y] = lmh_dict['min_values'][c]
#
#         softmax = nn.Softmax(dim=1)
#         predictions_vector = softmax(model(pert_img).data)
#         pred = torch.argmax(predictions_vector)
#         confidence = predictions_vector[0][img_y.item()].to(device)
#         if pred.item() != img_y.item():
#             return True, n_queries_pert, confidence, curr_k
#
#     return False, n_queries_pert, 0, curr_k
