import pickle
import program_
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.multiprocessing as tmp
import random
from torch.utils.data import DataLoader
from metropolis_hastings import run_MH
from utils import *


def run_program(program, model, dataloader, img_dim, center_matrix, max_queries, mean_norm,
                std_norm, device, amount_square, is_test=False, class_idx=None,
                results_path=None, max_k=1):
    """
    Run the specified program for adversarial attacks on a given model using the provided dataloader.

    Args:
        program (Program): A Program object containing the conditions for the adversarial attack.
        model (nn.Module): The neural network model to be attacked.
        dataloader (DataLoader): DataLoader containing the input images and labels.
        img_dim (int): The dimension of the input image (assuming a square image).
        center_matrix (torch.Tensor): A matrix representing the distance of each pixel to the image center. - REMOVE
        max_g (int): The maximum number of pixels to perturb with finer granularity.- REMOVE
        g (int): The level of granularity.- REMOVE
        lmh_dict (dict): A dictionary containing the 'min_values', 'mid_values', and 'max_values' for the perturbations.
        - FOR NOW WE NEED TO REMOVE IT
        mean_norm (list[float]):  The mean values for each channel used in image normalization.
        std_norm (list[float]):  The standard deviation values for each channel used in image normalization.
        device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').
        is_test (bool, optional): Whether the current run is for the test set. Defaults to False.
        class_idx (int, optional): Index of the current class. Required if is_test is True. Defaults to None.
        results_path (str, optional): Path to the directory where the results CSV file should be saved.
        max_k (int, optional): Maximal number of perturbed pixels. Defaults to 1.

    Returns:
        float: The average number of queries required to succeed in the adversarial attack.
    """
    model = model.to(device)
    model.eval()
    center_matrix = center_matrix.to(device)

    pert_img_to_idx_dict = create_pert_type_to_idx_dict()  # notice: pert per_square(0,0,0) -> 0, (1,0,0) -> 1 ect
    if is_test:
        results_df = pd.DataFrame(columns=["batch_idx", "class", "is_success", "queries", "pert_img"])
    num_imgs, num_success, sum_queries = 0, 0, 0
    max_queries = max_queries

    # go over the images data and +1 if we found correctly classified image
    for batch_idx, (data, target) in enumerate(dataloader):
        is_success = False
        img_x, img_y = data.to(device), target.to(device)
        if is_test:
            n_perturbed_pixels = 1
            if img_y.item() != class_idx:
                continue
        if not is_correct_prediction(model, img_x, img_y):
            continue
        num_imgs += 1
        possible_loc_pert_list = create_sorted_pert_list(amount_square)  # for now - it returns basic LIST not sorted!
        random.shuffle(possible_loc_pert_list)
        # perturbations list, each item contain 4 tuples because the image is split to 4 squares
        possible_loc_pert_list.append("STOP")
        # indicators_tensor = torch.zeros((8, img_dim, img_dim))
        orig_prob = get_orig_confidence(model, img_x, img_y, device)
        n_queries = 0
        pert_img_to_df = []
        min_prob_dict = {}
        # few_pixel_list = []
        all_pert_neighbors_list = create_neighbors_list()
        for pert_img in possible_loc_pert_list:  # pert_img is list with amount of squares tuple inside
            if (n_queries >= max_queries and is_test) or is_success:
                break

            if pert_img == "STOP":
                continue

            is_success, queries, curr_prob = try_perturb_img(model, img_x, img_y, pert_img, device, amount_square)

            n_queries += queries
            if is_success:
                pert_img_to_df = pert_img
                sum_queries += n_queries
                break

            if check_cond(program.cond_1, img_x, orig_prob, curr_prob, amount_square):
                for num_square in range(amount_square):
                    square_pert = pert_img[num_square]  # this is a tuple (-,-,-)

                    # push back all the neighbors of this current square
                    pert_idx = pert_img_to_idx_dict[square_pert]
                    curr_pert_neighbors_list = all_pert_neighbors_list[pert_idx]  # list of 3 tuple
                    for neighbor in curr_pert_neighbors_list:
                        closest_pert = create_new_neighbors_pert(neighbor, num_square, pert_img)  # this is a
                        # list of amount of squares tuples
                        possible_loc_pert_list.append(possible_loc_pert_list. \
                                                      pop(possible_loc_pert_list.index(closest_pert)))

            pert_queue = initialize_pixels_conf_queues(pert_img, curr_prob)  # pert_img is a
            # perturbation contain 4 tuples
            while (not pert_queue.empty()) and not is_success:
                if n_queries >= max_queries and is_test:
                    break

                pert_prob = pert_queue.get()  # this queue contains square perturbation
                curr_pert = pert_prob[0]
                curr_prob = pert_prob[1]
                # change to Queue of perturbation only !!

                if check_cond(program.cond_2, img_x, orig_prob, curr_prob, amount_square):

                    for num_square in range(amount_square):  # currently we have 4 squares for 1 image
                        if is_success:
                            break
                        square_pert = curr_pert[num_square]  # this is a tuple (-,-,-)
                        pert_idx = pert_img_to_idx_dict[square_pert]
                        curr_pert_neighbors_list = all_pert_neighbors_list[pert_idx]  # list of 3 tuple
                        # go over closest perturbation
                        for neighbor in curr_pert_neighbors_list:
                            # remove from all possible perturbation list
                            closest_pert = create_new_neighbors_pert(neighbor, num_square, curr_pert)
                            try:
                                indicator_idx = possible_loc_pert_list.index(closest_pert)
                            except:
                                continue

                            possible_loc_pert_list.pop(indicator_idx)

                            if n_queries >= max_queries and is_test:
                                break

                            is_success, queries, curr_prob = try_perturb_img(model, img_x, img_y, closest_pert,
                                                                             device, amount_square)
                            n_queries += queries
                            pert_queue.put((closest_pert, curr_prob))
                            if is_success:
                                pert_img_to_df = closest_pert
                                sum_queries += n_queries
                                break

        if is_success:
            num_success += 1

        if is_test:
            results_df = update_results_df(results_df, results_path, batch_idx, class_idx, is_success, n_queries,
                                           pert_img_to_df)
    model.to('cpu')
    if not is_test:
        return sum_queries / num_success


def synthesize(args):
    """
    Synthesizes a program for each specified class using the given arguments.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments, including:
            - args.model (str): The name of the model to use.
            - args.data_set (str): The name of the dataset to use.
            - args.classes_list (list[int]): A list of class indices to synthesize programs for.
            - args.num_train_images (int): The number of images in the training set.
            - args.max_iter (int): The maximum number of iterations for the MH algorithm.
            - args.num_iter_stop (int): The number of iterations without change before stopping.
            - args.g (int): The level of granularity.
            - args.max_g (int): The number of pixels with finer granularity.
            - args.max_queries (int) : The maximal number of possible queries per image.
            - args.mean_norm (list[float]):  The mean values for each channel used in image normalization.
            - args.std_norm (list[float]):  The standard deviation values for each channel used in image normalization.


    Returns:
        None

    Side Effects:
        - Saves the synthesized program for each class to a '.pkl' file.
        - Writes the synthesis results to a '.txt' file.
        - Prints the progress of the synthesis process.
    """
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Set up device(s)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = setup_devices()
    print(device)

    # Load train data
    train_data, img_dim = get_data_set(args)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=1)

    # Load model
    model = load_model(args.model)

    # Generate center matrix
    center_matrix = generate_center_matrix(img_dim)

    # Create low_mid_high_values dict
    lmh_dict = create_low_mid_high_values_dict(args.mean_norm, args.std_norm)

    # Write initial synthesis information to a file
    with open(args.model + '_' + args.data_set + '.txt', 'w') as f:
        f.write("data set: " + args.data_set + "\n")
        f.write("number of training images: " + str(args.num_train_images) + "\n")

    program_dict = {}

    # Iterate over the specified classes
    for class_idx in args.classes_list:
        print("########################")
        print("synthesizing program for class : ", class_idx)
        print("########################")

        # Select a subset of images from the class
        train_imgs_idx = select_n_images(args.num_train_images, class_idx, train_loader, model,
                                                                args.max_g, args.g,
                                                                lmh_dict, args.mean_norm, args.std_norm, device,
                                                                args.amount_square)
        n_train_data = torch.utils.data.Subset(train_data, train_imgs_idx)
        data_loader = torch.utils.data.DataLoader(n_train_data, shuffle=False, batch_size=1)

        # Initialize the best program and its query count
        best_program = program_.Program(img_dim)

        best_queries = run_program(best_program, model, data_loader, img_dim, center_matrix, \
                                   args.max_queries, args.mean_norm, args.std_norm, device, args.amount_square)
        previous_best_queries = None
        num_same_best_queries_iter = 1

        # Start the main synthesis loop
        with tqdm(total=args.max_iter, desc="Synthesizing program",
                  bar_format="{l_bar}{bar:10}{r_bar}") as pbar:

            # Perform synthesis in chunks
            for iter_idx in range(int(args.max_iter / 10)):

                # Update the iteration counter for unchanged query counts
                if previous_best_queries == best_queries:
                    num_same_best_queries_iter += 1
                else:
                    num_same_best_queries_iter = 1

                # Stop early if the query count hasn't changed for a few iterations
                if num_same_best_queries_iter == int(args.num_iter_stop / 10):
                    break

                # Update the previous best query count
                previous_best_queries = best_queries

                # Set up a queue and context for multiprocessing
                queue_proc = tmp.Manager().Queue()
                ctx = tmp.get_context('spawn')
                processes = []
                print("\nafter get context")
                # Create processes for each device
                for device_ in devices:
                    processes.append(ctx.Process(target=run_MH, \
                                                 args=(
                                                     best_program, best_queries, model, data_loader, img_dim,
                                                     center_matrix,
                                                     10, queue_proc, \
                                                     args.max_g, args.g, args.max_queries, lmh_dict, args.mean_norm,
                                                     args.std_norm, device_, args.amount_square
                                                     )))
                # Start and join all processes
                for proc in processes:
                    proc.start()
                for proc in processes:
                    proc.join()

                # Update the best program and its query count
                best_program, best_queries = update_best_program(queue_proc, best_program, best_queries)
                pbar.update(10)
            program_dict[class_idx] = best_program
            pickle.dump(program_dict, open(args.model + '_' + args.data_set + '.pkl', 'wb'))
            write_program_results(args, class_idx, best_program, best_queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OPPSLA Synthesizer')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='Model architecture to use (e.g., vgg16, resnet18, etc.)')
    parser.add_argument('--data_set', default='cifar10', type=str, choices=['cifar10', 'imagenet'],
                        help='Dataset to use - must be CIFAR-10 or ImageNet')
    parser.add_argument('--classes_list', default=list([1]), metavar='N', type=int,
                        nargs='+', help='List of classes for the synthesis')
    parser.add_argument('--num_train_images', default=10, type=int, help='# of images in the training set per class')
    parser.add_argument('--imagenet_dir', type=str, help='Directory containing ImageNet dataset images')
    parser.add_argument('--max_iter', default=210, type=int, help='Maximum # of iterations for the MH algorithm')
    parser.add_argument('--num_iter_stop', default=60, type=int,
                        help='# of iterations without change before stopping the algorithm')
    parser.add_argument('--g', default=0, type=int, help='Granularity level for the synthesis process')
    parser.add_argument('--max_g', default=0, type=int, help='Maximum number of pixels with finer granularity')
    parser.add_argument('--max_queries', default=500, type=int, help='maximal number of queries per image')
    parser.add_argument('--mean_norm', metavar='N', type=float, nargs='+', default=[0.0, 0.0, 0.0], \
                        help='List of mean values for each channel used in image normalization. Default is [0.0, 0.0, 0.0]')
    parser.add_argument('--std_norm', metavar='N', type=float, nargs='+', default=[1.0, 1.0, 1.0], \
                        help='List of standard deviation values for each channel used in image normalization. Default is [1.0, 1.0, 1.0]')
    parser.add_argument('--amount_square', default=6, type=int,
                        help='The number of squares that will divided the image of')

    args = parser.parse_args()
    synthesize(args)
