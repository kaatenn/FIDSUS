import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverproto import FedProto
from flcore.servers.servermoon import MOON
from flcore.servers.servergpfl import GPFL
from flcore.servers.servergh import FedGH
from flcore.servers.serveravgDBE import FedAvgDBE
from flcore.servers.FIDSUS import FIDSUS
from flcore.trainmodel.models import *
from utils.result_utils import average_data
from utils.config_loader import load_config



logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

emb_dim=32
torch.manual_seed(10)
def run(args):

    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model

        if model_str == "1dcnn":
            args.model = CNN1D(hidden_dim=emb_dim, num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedAvgDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvgDBE(args, i)

        elif args.algorithm == "FIDSUS":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FIDSUS(args, i)

        else:
            raise NotImplementedError

        server.train()

        # Save predictions for family-level evaluation (non-invasive)
        try:
            if hasattr(server, 'save_predictions_personalized'):
                server.save_predictions_personalized()
            else:
                server.save_predictions()
        except Exception as e:
            print(f"[WARN] Could not save predictions for family eval: {e}")

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    savetime_dir = 'timecost'
    os.makedirs(savetime_dir, exist_ok=True)
    timefilename = os.path.join(savetime_dir, f"time_cost_{args.algorithm}_{args.num_clients}_{args.dataset}_{args.goal}.txt")
    with open(timefilename, 'w') as file:
        file.write(f"Total time cost: {round(np.average(time_list), 2)}s.\n")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser(description="FIDSUS Federated Learning")
    parser.add_argument("--config", "-c", type=str, default="experiments/default.json",
                        help="Path to JSON experiment config file")
    cli_args = parser.parse_args()

    experiments = load_config(cli_args.config)

    for idx, args in enumerate(experiments):
        exp_name = getattr(args, 'name', f"{args.algorithm}_{args.dataset}_{args.goal}")
        if getattr(args, 'ignore', False):
            print(f"\n{'#' * 60}")
            print(f"# Experiment {idx+1}/{len(experiments)}: {exp_name} [SKIPPED]")
            print(f"{'#' * 60}")
            continue

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        if args.device == "cuda" and not torch.cuda.is_available():
            print("\ncuda is not avaiable.\n")
            args.device = "cpu"
        print("=" * 50)
        print("Algorithm: {}".format(args.algorithm))
        print("Local batch size: {}".format(args.batch_size))
        print("Goal: {}".format(args.goal))
        print("Local epochs: {}".format(args.local_epochs))
        print("Local learing rate: {}".format(args.local_learning_rate))
        print("Local learing rate decay: {}".format(args.learning_rate_decay))
        if args.learning_rate_decay:
            print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
        print("Total number of clients: {}".format(args.num_clients))
        print("Clients join in each round: {}".format(args.join_ratio))
        print("Clients randomly join: {}".format(args.random_join_ratio))
        print("Client activity rate: {}".format(args.client_activity_rate))

        if args.device == "cuda":
            print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("=" * 50)

        run(args)

    print(f"\nAll {len(experiments)} experiment(s) completed.")
    print(f"Total elapsed: {time.time() - total_start:.1f}s")
