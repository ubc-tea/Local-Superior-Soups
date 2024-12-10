#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE

from flcore.servers.serversoup import FedSoup

from flcore.servers.serverlocalsuperiorsoups import LocalSuperiorSoups

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import BiLSTM_TextClassification

# from flcore.trainmodel.resnet import resnet18 as resnet
from flcore.trainmodel.alexnet import alexnet
from flcore.trainmodel.mobilenet_v2 import mobilenet_v2

from flcore.trainmodel.prefix_tuning_model import PrefixTuningModel

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

import timm

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    print("All arguments: ", args)

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if args.no_pretrained:
            pretrained = False
        else:
            pretrained = True

        # Generate args.model
        if model_str == "mlr":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = Mclr_Logistic(
                    1 * 28 * 28, num_classes=args.num_classes
                ).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = Mclr_Logistic(
                    3 * 32 * 32, num_classes=args.num_classes
                ).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(
                    args.device
                )

        elif model_str == "cnn":
            if args.dataset[:5] == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=1024
                ).to(args.device)
            elif args.dataset == "omniglot":
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=33856
                ).to(args.device)
            elif args.dataset[:5] == "Cifar":
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=1600
                ).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif args.dataset == "Digit5":
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=10816
                ).to(args.device)

        elif model_str == "dnn":  # non-convex
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(
                    args.device
                )
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(
                    args.device
                )
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(
                pretrained=pretrained, num_classes=args.num_classes
            ).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(
                pretrained=pretrained, num_classes=args.num_classes
            ).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(
                pretrained=pretrained,
                aux_logits=False,
                num_classes=args.num_classes,
            ).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(
                pretrained=pretrained, num_classes=args.num_classes
            ).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(
                input_size=vocab_size,
                hidden_size=hidden_dim,
                output_size=args.num_classes,
                num_layers=1,
                embedding_dropout=0,
                lstm_dropout=0,
                attention_dropout=0,
                embedding_length=hidden_dim,
            ).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(
                hidden_dim=hidden_dim,
                max_len=max_len,
                vocab_size=vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(
                ntoken=vocab_size,
                d_model=hidden_dim,
                nhead=2,
                d_hid=hidden_dim,
                nlayers=2,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        else:
            if model_str in timm.list_models():
                # evaluated vgg16, resnet18
                args.model = timm.create_model(
                    model_str, pretrained=pretrained, num_classes=args.num_classes
                ).to(args.device)

                # tmp: freezing backbone
                for name, param in args.model.named_parameters():
                    if "fc" not in name:
                        print("freezing except for fc layers ", name)
                        param.requires_grad = False

                if args.prefix_tuning:
                    args.model = PrefixTuningModel(base=args.model, prefix_type="VAE").to(args.device)

            else:
                raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedSoup":
            server = FedSoup(args, i)

        elif args.algorithm == "LocalSuperiorSoups":
            server = LocalSuperiorSoups(args, i)

        else:
            raise NotImplementedError

        if args.test_time_adaptation_eval:
            server.tta_eval()
        else:
            server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(
        dataset=args.dataset,
        algorithm=args.algorithm,
        goal=args.goal,
        times=args.times,
        length=args.global_rounds / args.eval_gap + 1,
    )

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="mnist")
    parser.add_argument("-nb", "--num_classes", type=int, default=2)
    parser.add_argument("-m", "--model", type=str, default="cnn")
    parser.add_argument("-p", "--head", type=str, default="cnn")
    parser.add_argument("-lbs", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.005,
        help="Local learning rate",
    )
    parser.add_argument("-gr", "--global_rounds", type=int, default=1)
    parser.add_argument("-ls", "--local_steps", type=int, default=8)
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        action="store_true",
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=3, help="Total number of clients"
    )
    parser.add_argument(
        "-pv", "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument(
        "-leg",
        "--local_eval_gap",
        type=int,
        default=1,
        help="Local step gap for evaluation",
    )
    parser.add_argument(
        "-dp", "--privacy", action="store_true", default=False, help="differential privacy"
    )
    parser.add_argument("-dps", "--dp_sigma", type=float, default=0.0)
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="models")
    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when training locally",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        action="store_true",
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument(
        "-bt",
        "--beta",
        type=float,
        default=0.0,
        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer",
    )
    parser.add_argument(
        "-lam",
        "--lamda",
        type=float,
        default=1.0,
        help="Regularization weight for pFedMe and FedAMP",
    )
    parser.add_argument(
        "-mu", "--mu", type=float, default=0, help="Proximal rate for FedProx"
    )
    parser.add_argument(
        "-K",
        "--K",
        type=int,
        default=5,
        help="Number of personalized training steps for pFedMe",
    )
    parser.add_argument(
        "-lrp",
        "--p_learning_rate",
        type=float,
        default=0.01,
        help="personalized learning rate to caculate theta aproximately using K steps",
    )
    # FedFomo
    parser.add_argument(
        "-M",
        "--M",
        type=int,
        default=5,
        help="Server only sends M client models to one client at each round",
    )
    # FedMTL
    parser.add_argument(
        "-itk",
        "--itk",
        type=int,
        default=4000,
        help="The iterations for solving quadratic subproblems",
    )
    # FedAMP
    parser.add_argument(
        "-alk",
        "--alphaK",
        type=float,
        default=1.0,
        help="lambda/sqrt(GLOABL-ITRATION) according to the paper",
    )
    parser.add_argument("-sg", "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument("-al", "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument("-pls", "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument("-ta", "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument("-fts", "--fine_tuning_steps", type=int, default=1)
    # APPLE
    parser.add_argument("-dlr", "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument("-L", "--L", type=float, default=1.0)

    # learning rate decay
    parser.add_argument("-lrdecay", "--learning_rate_decay", type=list, default=[])

    # FedSoup soup wa alpha
    parser.add_argument(
        "-wa_alpha",
        "--wa_alpha",
        type=float,
        default=0.75,
        help="Weight averaging ratio of personalized global model pool for FedSoup",
    )

    # for out-of-federation evaluation
    parser.add_argument(
        "-hoid",
        "--hold_out_id",
        type=int,
        default=1e8,
        help="Hold-out out-of-federated evaluation set. 1e8 means no hold-out set",
    )

    # add loading pre-trained model
    parser.add_argument(
        "-ptm_path",
        "--pretrain_model_path",
        type=str,
        default="",
        help="Pre-trained model path for initial model loading",
    )
    parser.add_argument(
        "-mon_hessian",
        "--monitor_hessian",
        action="store_true",
        default=False,
        help="Monitoring hessian eigen during training",
    )
    parser.add_argument(
        "-tta",
        "--test_time_adaptation",
        action="store_true",
        default=False,
        help="Test-Time adaptation for out-of-fedration data.",
    )
    parser.add_argument(
        "-tta_eval",
        "--test_time_adaptation_eval",
        action="store_true",
        default=False,
        help="Only Test-Time Adpatation on out-of-federation data (no training).",
    )
    parser.add_argument(
        "--save_img",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=False,
        help="Pretrained model on public large scale dataset, e.g., ImageNet.",
    )

    parser.add_argument(
        "--server_training",
        action="store_true",
        default=False,
        help="Training on the server side with public dataset.",
    )

    parser.add_argument(
        "--server_dataset",
        type=str,
        default="global",
        help="Current support options: global or public. Global dataset is sampled from each local dataset. Public dataset is ImageNet or TinyImageNet for image classification model.",
    )

    parser.add_argument(
        "--server_global_dataset_rate",
        type=float,
        default=1e-3,
        help="Sever global dataset rate of the sum of all local dataset",
    )

    parser.add_argument(
        "--server_gen_data",
        action="store_true",
        default=False,
        help="Genrating synthetic data from local trained model or not.",
    )

    parser.add_argument(
        "--n_verts",
        type=int,
        default=2,
        help="number of vertices in a soups model (total ensembling n_verts + 1 models)",
    )

    parser.add_argument(
        "--div_coeff",
        type=float,
        default=1e-3,
        help="Coefficient of volume loss of a soups model",
    )

    parser.add_argument(
        "--aff_coeff",
        type=float,
        default=1e-3,
        help="Coefficient of distance from initialization",
    )

    parser.add_argument(
        "--linear_probing",
        action="store_true",
        default=False,
        help="Linear Probing before all layer fine-tuning.",
    )

    parser.add_argument(
        "--linear_probing_steps", type=int, default=160, help="Linear Probing steps."
    )

    parser.add_argument(
        "--connecting_coeff",
        type=float,
        default=0.5,
        help="Connecting different solutions in a low-loss basin.",
    )

    parser.add_argument(
        "--n_soups_sampling",
        type=int,
        default=2,
        help="number of model sampling to use per iteration in a soups model",
    )

    parser.add_argument("--no_soups_fc", action="store_true", default=False, help="soups parameters exclude fc linear layers.")

    parser.add_argument("--used_imp", action="store_true", default=False, help="not use fisher-rao distance.")

    parser.add_argument("--global_steps", type=int, default=4)

    # cmh note: only support for pretrained currently
    parser.add_argument("--prefix_tuning", action="store_true", default=False, help="adding prefix layer for fine-tuning.")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Hold-out Client ID: {}".format(args.hold_out_id))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)