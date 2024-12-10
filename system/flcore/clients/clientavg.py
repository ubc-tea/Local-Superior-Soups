import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

from torchcontrib.optim import SWA

import copy


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.learning_rate_decay = args.learning_rate_decay
        self.local_eval_gap = args.local_eval_gap

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # base_opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = SWA(base_opt, swa_start=16, swa_freq=2, swa_lr=self.learning_rate * 0.1)

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )

        start_time = time.time()

        max_local_steps = self.local_steps

        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer.zero_grad()
                # base_opt.zero_grad()

                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

            if step % self.local_eval_gap == 0:
                print("Current Local Step: {}".format(step))
                # tmp remove ood_eval
                # self.local_evaluate(ood_eval=True, global_eval=True)
                self.local_evaluate(ood_eval=False, global_eval=False)

        # self.model.cpu()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    # # FedBN when having BN layers
    # def set_parameters(self, model, direct_copy=False):
    #     if direct_copy:
    #         print("Direct copying model from server")
    #         self.model = copy.deepcopy(model)

    #     for (nn, np), (on, op) in zip(model.named_parameters(), self.model.named_parameters()):
    #         if 'bn' not in nn:
    #             op.data = np.data.clone()
