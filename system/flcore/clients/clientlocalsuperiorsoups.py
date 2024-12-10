import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

from utils.soups.superior_soups import SuperiorSoups

import copy


class clientLocalSuperiorSoups(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()

        self.learning_rate_decay = args.learning_rate_decay
        self.local_eval_gap = args.local_eval_gap

        self.n_verts = args.n_verts
        self.div_coeff = args.div_coeff
        self.aff_coeff = args.aff_coeff
        self.n_soups_samping = args.n_soups_sampling

        self.linear_probing = args.linear_probing
        self.no_soups_fc = args.no_soups_fc

        self.used_imp = args.used_imp

    def train(self):
        trainloader = self.load_train_data()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )

        start_time = time.time()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # linear probing
        # if self.linear_probing:
        #     self.linear_probing_training()

        self.init_model = copy.deepcopy(self.model)
        self.init_model.load_state_dict(self.model.state_dict())

        # pre-computing parameter importance
        if not self.used_imp:
            # reset init_model
            self.init_model.load_state_dict(self.model.state_dict())

            soups_model = SuperiorSoups(
                self.model, num_vertices=1, fixed_points=[False], no_soups_fc=self.no_soups_fc
            ).to(self.device)

            self.model = soups_model

        else:
            print("\nPre-computing parameter importance......")
            self.grad_model = copy.deepcopy(self.model)

            for g_p in self.grad_model.parameters():
                g_p.data.zero_()

            self.init_model.train()
            self.optimizer = torch.optim.Adam(
                self.init_model.parameters(), lr=0.1
            )
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                output = self.init_model(x)
                loss = self.loss(output, y)

                loss.backward()

                for p, g_p in zip(self.init_model.parameters(), self.grad_model.parameters()):
                    if p.grad is not None:
                        g_p.data += p.grad.pow(2).clone()
            
            for g_p in self.grad_model.parameters():
                g_p.data /= len(trainloader)
                g_p.data = torch.clamp(g_p.data.clone(), min=1e-8, max=1e8)
                g_p.data = g_p.data.sqrt()
                # print("g_p value: ", g_p.data)

            # reset init_model
            self.init_model.load_state_dict(self.model.state_dict())

            soups_model = SuperiorSoups(
                self.model, num_vertices=1, fixed_points=[False], no_soups_fc=self.no_soups_fc
            ).to(self.device)
            soups_model.add_weight_importance(self.grad_model)

            # print("init soups model: ", soups_model.state_dict().keys())

            self.model = soups_model

            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            print("\nCompleted computing parameter importance......")


        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.local_steps
        )

        self.model.train()
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
                aff_loss = self.aff_coeff * self.model.dist_from_init()
                loss += aff_loss

                loss.backward()
                self.optimizer.step()

            # self.scheduler.step()

            if step % self.local_eval_gap == 0:
                print("Loss: ", loss.item())
                print("Affinity Loss: ", aff_loss.item())
                print("Current Local Step: {}".format(step))
        
        self.local_evaluate(ood_eval=False, global_eval=True)

        print("Training with Diversity Loss......")

        grad_accumulation_step = 15

        fixed_flag_list = [True]
        for vv in range(1, self.n_verts + 1):
            fixed_flag_list[-1] = True
            fixed_flag_list.append(False)

            print("Adding a new model")
            self.model.add_vert()
            self.model._fix_points(fixed_flag_list)

            self.model = self.model.to(self.device)
            self.model.train()

            # self.optimizer = torch.optim.SGD(
            #     self.model.parameters(), lr=self.learning_rate
            # )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.local_steps
            )

            # half of fine-tuning from scratch
            for step in range(max_local_steps // 8):
                start = time.time()
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    self.optimizer.zero_grad()

                    loss = 0.0
                    # n soups vetices sampling
                    for _ in range(self.n_soups_samping):
                        output = self.model(x)
                        loss += self.loss(output, y)
                        
                    loss.div(self.n_soups_samping)

                    # accumulate grad for fast training
                    loss = loss / grad_accumulation_step

                    # diversity loss maximization
                    aff_loss = self.aff_coeff * self.model.dist_from_init()

                    vol = self.model.total_volume()
                    div_loss = self.div_coeff * vol

                    loss = loss - div_loss + aff_loss

                    loss.backward()

                    if (i + 1) % grad_accumulation_step == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        print("Loss: ", loss.item())
                        print("Diveristy Loss: ", div_loss.item())
                        print("Affinity Loss: ", aff_loss.item())

                end = time.time()
                print("Costing time: ", end - start)

                # self.scheduler.step()

                if step % self.local_eval_gap == 0:
                    print("Current Local Step: {}".format(step))
                    self.local_evaluate(ood_eval=False, global_eval=True)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
        
        self.model.unsoups() 

        trained_state_dict = self.model.base.state_dict()

        self.init_model.load_state_dict(trained_state_dict)

        self.model = copy.deepcopy(self.init_model)
        self.model.load_state_dict(self.init_model.state_dict())

        print("Local evaluating after unsoups......")
        self.local_evaluate(ood_eval=False, global_eval=True)
