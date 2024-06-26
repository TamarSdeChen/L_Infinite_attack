import pickle
import program_
import argparse
import torch.multiprocessing as tmp
from synthesize import run_program
from utils import *


def attack(args):
    """
    Perform the adversarial attack using the provided settings and programs.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.
            - args.model (str): The name of the pre-trained model to use for classification.
            - args.program_path (str): The path to the pickled file containing the synthesized programs.
            - args.results_path (str): Path to the directory where the results CSV file should be saved.
            - args.classes_list (List[int]): The list of classes to attack.
            - args.max_g (int): The maximum number of pixels to perturb with finer granularity.
            - args.g (int): The level of granularity.
            - args.max_k (int): Maximum number of pixels that can be perturbed.
            - args.max_queries (int) : The maximal number of possible queries per image.
            - args.mean_norm (list[float]):  The mean values for each channel used in image normalization.
            - args.std_norm (list[float]):  The standard deviation values for each channel used in image normalization.
    """
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Set up device(s)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # devices = setup_devices()

    # Load test data
    test_data, img_dim = get_data_set(args, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

    # Load model
    model = load_model(args.model)
    model = model.to('cpu')

    # Load pre-synthesized programs
    program_dict = pickle.load(open(args.program_path, 'rb'))

    # Generate center matrix
    center_matrix = generate_center_matrix(img_dim)

    # Create low_mid_high_values dict
    lmh_dict = create_low_mid_high_values_dict(args.mean_norm, args.std_norm)

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    run_program(program_dict[args.classes_list[0]], model, test_loader, img_dim, center_matrix, args.max_queries,
                args.mean_norm, args.std_norm, device, args.amount_square, True, args.classes_list[0],
                args.results_path, args.max_k)


if __name__ == '__main__':
    tmp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='OPPSLA attack')
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('--data_set', default='cifar10', type=str, help='data set - must be CIFAR-10 or ImageNet')
    parser.add_argument('--classes_list', default=list([1]), metavar='N', type=int, nargs='+',
                        help='classes for the synthesis process')
    parser.add_argument('--imagenet_dir', type=str, help='directory for images of ImageNet dataset')
    parser.add_argument('--program_path', default='resnet18_cifar10.pkl', type=str,
                        help='path of the program as a pkl file')
    parser.add_argument('--results_path', default="./results_L_inifinity", type=str, help='path of the saved results')
    parser.add_argument('--g', default=0, type=int, help='level of granularity')
    parser.add_argument('--max_g', default=0, type=int, help='number of pixels with finer granularity')
    parser.add_argument('--max_queries', default=10000, type=int, help='maximal number of queries per image')
    parser.add_argument('--max_k', default=1, type=int, help='maximal number of perturbed pixels')
    parser.add_argument('--mean_norm', metavar='N', type=float, nargs='+', default=[0.0, 0.0, 0.0], \
                        help='List of mean values for each channel used in image normalization. Default is [0.0, 0.0, 0.0]')
    parser.add_argument('--std_norm', metavar='N', type=float, nargs='+', default=[1.0, 1.0, 1.0], \
                        help='List of standard deviation values for each channel used in image normalization. Default is [1.0, 1.0, 1.0]')
    parser.add_argument('--amount_square', default=4, type=int,
                        help='The number of squares that will divided the image of')

    args = parser.parse_args()
    attack(args)
