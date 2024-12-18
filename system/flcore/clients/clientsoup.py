import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

import copy


class clientSoup(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.learning_rate_decay = args.learning_rate_decay

        self.wa_alpha = args.wa_alpha
        self.per_global_model_num = 0
        self.per_global_model = None
        self.last_global_model = None
        self.wa_model = None
        self.update_wa_model = None
        self.train_round = 0
        self.tot_round = args.global_rounds
        self.id = id

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )

        start_time = time.time()

        self.last_global_model = copy.deepcopy(self.model)
        self.last_global_model.load_state_dict(self.model.state_dict())

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        self.train_round += 1
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
                output = self.model(x)
                loss = self.loss(output, y)

                # try to add volume loss
                model_list = [self.last_global_model, self.model]
                vol_loss = self.volume_loss(model_list)

                loss += 1e-5 * vol_loss

                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        if self.train_round > self.wa_alpha * self.tot_round:
            print("Begin Weight Averaging......")

            if self.per_global_model_num == 0:
                self.per_global_model = copy.deepcopy(self.model)
                self.per_global_model.load_state_dict(
                    self.last_global_model.state_dict()
                )

            self.wa_model = copy.deepcopy(self.model)
            self.update_wa_model = copy.deepcopy(self.model)

            self.wa_model.load_state_dict(self.model.state_dict())
            self.update_wa_model.load_state_dict(self.model.state_dict())

            for wa_param, u_wa_param, global_param, last_global_model in zip(
                self.wa_model.parameters(),
                self.update_wa_model.parameters(),
                self.per_global_model.parameters(),
                self.last_global_model.parameters(),
            ):
                wa_param.data = wa_param.data.clone() * (
                    1.0 / (self.per_global_model_num + 1.0)
                ) + global_param.data.clone() * (
                    self.per_global_model_num / (self.per_global_model_num + 1.0)
                )
                u_wa_param.data = (
                    u_wa_param.data.clone() * (1.0 / (self.per_global_model_num + 2.0))
                    + global_param.data.clone()
                    * (self.per_global_model_num / (self.per_global_model_num + 2.0))
                    + last_global_model.data.clone()
                    * (1.0 / (self.per_global_model_num + 2.0))
                )
                # preparing for updated per_global_model
                last_global_model.data = (1.0 / (self.per_global_model_num + 1.0)) * (
                    self.per_global_model_num * global_param.data.clone()
                    + last_global_model.data.clone()
                )

            # local_acc = self.quick_test(self.model)
            wa_acc = self.quick_test(self.wa_model)
            update_wa_acc = self.quick_test(self.update_wa_model)
            # print("Local Accuracy: ", local_acc)
            print("Original Weight Averaging Accuracy: ", wa_acc)
            print("Updated Weight Averaging Accuracy: ", update_wa_acc)

            if update_wa_acc > wa_acc:
                print("Update Personalized Global Model......")
                self.model.load_state_dict(self.update_wa_model.state_dict())
                self.per_global_model.load_state_dict(
                    self.last_global_model.state_dict()
                )
                self.per_global_model_num += 1
                print("Client ID: ", self.id)
                print("Personalized Global Model Num: ", self.per_global_model_num)
            else:
                print("Remain the same Personalized Global Model.")
                self.model.load_state_dict(self.wa_model.state_dict())
            del self.last_global_model, self.wa_model, self.update_wa_model

            # # forced interpolation of local and global
            # if self.per_global_model_num == 0:
            #     self.per_global_model_num += 1
            #     self.per_global_model = copy.deepcopy(self.model)
            #     self.per_global_model.load_state_dict(self.last_global_model.state_dict())
            #     for param, per_global_param in zip(self.model.parameters(), self.per_global_model.parameters()):
            #         param.data = param.data * 0.5 + per_global_param.data * 0.5
            #         # param.data = param.data * (1 - self.wa_alpha) + per_global_param.data * self.wa_alpha
            # else:
            #     self.wa_model = copy.deepcopy(self.model)
            #     self.update_wa_model = copy.deepcopy(self.model)
            #     self.wa_model.load_state_dict(self.model.state_dict())
            #     self.update_wa_model.load_state_dict(self.model.state_dict())

            #     for wa_param, u_wa_param, global_param, last_global_model in zip(self.wa_model.parameters(), self.update_wa_model.parameters(), self.per_global_model.parameters(), self.last_global_model.parameters()):
            #         wa_param.data = wa_param.data.clone() * (1.0 / (self.per_global_model_num + 1.0)) + global_param.data.clone() * (self.per_global_model_num / (self.per_global_model_num + 1.0))
            #         u_wa_param.data = u_wa_param.data.clone() * (1.0 / (self.per_global_model_num + 2.0)) +  global_param.data.clone() * (self.per_global_model_num / (self.per_global_model_num + 2.0)) + last_global_model.data.clone() * (1.0 / (self.per_global_model_num + 2.0))

            #         # wa_param.data = wa_param.data.clone() * (1 - self.wa_alpha) + global_param.data.clone() * self.wa_alpha
            #         # u_wa_param.data = u_wa_param.data.clone() * (1 - self.wa_alpha) + (1.0 / (self.per_global_model_num + 1.0)) * (self.per_global_model_num * global_param.data.clone() + last_global_model.data.clone()) * self.wa_alpha

            #         # preparing for updated per_global_model
            #         last_global_model.data = (1.0 / (self.per_global_model_num + 1.0)) * (self.per_global_model_num * global_param.data.clone() + last_global_model.data.clone())

            #     wa_acc = self.quick_test(self.wa_model)
            #     update_wa_acc = self.quick_test(self.update_wa_model)

            #     print("Original Weight Averaging Accuracy: ", wa_acc)
            #     print("Updated Weight Averaging Accuracy: ", update_wa_acc)

            #     if update_wa_acc > wa_acc:
            #         print("Update Personalized Global Model......")
            #         self.model.load_state_dict(self.update_wa_model.state_dict())
            #         self.per_global_model.load_state_dict(self.last_global_model.state_dict())
            #         self.per_global_model_num += 1
            #         print("Client ID: ", self.id)
            #         print("Personalized Global Model Num: ", self.per_global_model_num)
            #     else:
            #         print("Remain the same Personalized Global Model.")
            #         self.model.load_state_dict(self.wa_model.state_dict())
            #     del self.last_global_model, self.wa_model, self.update_wa_model

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def quick_test(self, model):
        testloaderfull = self.load_test_data()
        model.to(self.device)
        model.eval()

        test_acc_count = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc_count += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc_count / test_num
