import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from utils.data_utils import read_client_data

import copy

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.global_init_model = copy.deepcopy(args.model)
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.monitor_hessian = args.monitor_hessian
        self.test_time_adaptation = args.test_time_adaptation
        self.test_time_adaptation_eval = args.test_time_adaptation_eval
        self.hold_out_id = args.hold_out_id
        if self.hold_out_id == 1e8:
            self.num_clients = args.num_clients
        else:
            self.num_clients = args.num_clients + 1
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.global_steps = args.global_steps

        self.server_global_dataset_rate = args.server_global_dataset_rate

        if args.no_pretrained:
            self.pretrained = False
        else:
            self.pretrained = True

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(
            range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            if i == args.hold_out_id:
                continue
            print("set client: ", i)

            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(
                args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * (self.num_clients)))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(
                range(self.join_clients, self.num_clients), 1, replace=False
            )[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(
            np.random.choice(self.clients, join_clients, replace=False)
        )

        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            # client.set_parameters(self.global_model, direct_copy=True)
            client.set_parameters(self.global_model)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients)
        )

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost["total_cost"] / (
                client.train_time_cost["num_rounds"] + 1e-12
            ) + client.send_time_cost["total_cost"] / (
                client.send_time_cost["num_rounds"] + 1e-12
            )
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        # averaging state_dict()
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        global_state_dict = self.uploaded_models[0].state_dict()
        for key in global_state_dict:
            for idx, client_model in enumerate(self.uploaded_models):
                if idx == 0:
                    # to align dtype
                    global_state_dict[key].data = self.uploaded_weights[idx] * client_model.state_dict()[key].data.clone()
                else:
                    global_state_dict[key].data += self.uploaded_weights[idx] *  client_model.state_dict()[key].data.clone()
        self.global_model.load_state_dict(global_state_dict)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(
            self.global_model.parameters(), client_model.parameters()
        ):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert os.path.exists(model_path)
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(
            item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def load_item(self, item_name):
        return torch.load(
            os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def test_metrics(self, ood_eval=False, global_eval=False):
        if ood_eval:
            # cmh NOTE: clients may be shuffled sometimes
            model_ids = []
            num_samples = []
            tot_correct = []
            tot_auc = []

            oof_model_ids = []
            oof_num_samples = []
            oof_tot_correct = []
            oof_tot_auc = []

            for c in self.clients:
                dataset_ids = []
                for dataset_id in range(self.num_clients):
                    if global_eval:
                        dataset_ids.append(dataset_id)
                    else:
                        if c.id != dataset_id:
                            dataset_ids.append(dataset_id)

                if self.hold_out_id == 1e8:
                    ct, ns, auc = c.test_metrics(
                        ood_eval=ood_eval, dataset_ids=dataset_ids
                    )
                else:
                    ct, ns, auc, oof_ct, oof_ns, oof_auc = c.test_metrics(
                        ood_eval=ood_eval, dataset_ids=[self.hold_out_id]
                    )
                    oof_model_ids.append(c.id)
                    oof_tot_correct.append(oof_ct * 1.0)
                    oof_tot_auc.append(oof_auc * oof_ns)
                    oof_num_samples.append(oof_ns)

                model_ids.append(c.id)
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)

            if self.hold_out_id == 1e8:
                return model_ids, num_samples, tot_correct, tot_auc

            return (
                model_ids,
                num_samples,
                tot_correct,
                tot_auc,
                oof_model_ids,
                oof_num_samples,
                oof_tot_correct,
                oof_tot_auc,
            )
            # for dataset_id in dataset_id_list:
            #     num_samples = []
            #     tot_correct = []
            #     tot_auc = []
            #     for c in self.clients:
            #         ct, ns, auc = c.test_metrics(ood_eval, dataset_id)

            #         tot_correct.append(ct*1.0)
            #         tot_auc.append(auc*ns)
            #         num_samples.append(ns)
            #     ids = [c.id for c in self.clients]
            #     total_list.append([ids, num_samples, tot_correct, tot_auc])
            # return total_list

        else:
            num_samples = []
            tot_correct = []
            tot_auc = []
            for c in self.clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)

            ids = [c.id for c in self.clients]
            return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None, ood_eval=False, global_eval=True, tuned_mode=""):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

        # cmh: added per client performance
        test_acc_list = []
        test_auc_list = []

        print(stats)
        print(stats[0])

        for idx, c_id in enumerate(stats[0]):
            test_acc_list.append(stats[2][idx] / stats[1][idx])
            test_auc_list.append(stats[3][idx] / stats[1][idx])

        for idx, c_id in enumerate(stats[0]):
            print("Client {} Test Accurancy: {:.4f}".format(c_id, test_acc_list[idx]))
            print("Client {} Test AUC: {:.4f}".format(c_id, test_auc_list[idx]))

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        # if acc is None or len(acc) == 0:
        #     for c in self.clients:
        #         c.best_model_dict = copy.deepcopy(c.model.state_dict())

        #     self.rs_test_acc.append(test_acc)
        # else:
        #     # save best history model
        #     if test_acc > max(acc):
        #         for c in self.clients:
        #             c.best_model_dict = copy.deepcopy(c.model.state_dict())
        #     acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if ood_eval:
            print("OOD eval details: ")
            # model_ids, num_samples, tot_correct, tot_auc = self.test_metrics(ood_eval=True)

            # id_tot_correct_list = tot_correct[:self.num_clients]
            # id_tot_auc_list = tot_auc[:self.num_clients]
            # id_num_samples_list = num_samples[:self.num_clients]

            # ood_tot_correct_list = tot_correct[self.num_clients:]
            # ood_tot_auc_list = tot_auc[self.num_clients:]
            # ood_num_samples_list = num_samples[self.num_clients:]

            stats_all = self.test_metrics(ood_eval=True)
            stats = stats_all[:4]

            print(stats)
            print(stats[0])

            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

            # cmh: added per client performance
            test_acc_list = []
            test_auc_list = []

            for idx, c_id in enumerate(stats[0]):
                test_acc_list.append(stats[2][idx] / stats[1][idx])
                test_auc_list.append(stats[3][idx] / stats[1][idx])

            for idx, c_id in enumerate(stats[0]):
                print(
                    "Client {} Test Accurancy: {:.4f}".format(c_id, test_acc_list[idx])
                )
                print("Client {} Test AUC: {:.4f}".format(c_id, test_auc_list[idx]))

            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
            accs = [a / n for a, n in zip(stats[2], stats[1])]
            aucs = [a / n for a, n in zip(stats[3], stats[1])]

            print("OOD Performance (client model i on all other dataset j):")
            print("OOD Client-Num-Weighted Test Accurancy: {:.4f}".format(test_acc))
            print("OOD Client-Num-Weighted Test AUC: {:.4f}".format(test_auc))

            print("OOD Client-Equally Test Accurancy: {:.4f}".format(np.mean(accs)))
            print("OOD Client-Equally Test AUC: {:.4f}".format(np.mean(aucs)))

            print("OOD Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("OOD Std Test AUC: {:.4f}".format(np.std(aucs)))

            # out-of-federated performance
            if self.hold_out_id != 1e8:
                stats = stats_all[4:]
                print(stats)
                print(stats[0])

                test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
                test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

                print("=" * 16)
                print("OOF Performance:")
                print("OOF Client Test Accurancy: {:.4f}".format(test_acc))
                print("OOF Client Test AUC: {:.4f}".format(test_auc))

        if ood_eval:
            print("\nGlobal Evaluating......")
            self.clients[0].local_evaluate(ood_eval=True, global_eval=True)

            if tuned_mode == "linear_probing":
                for cid in range(len(self.clients)):
                    self.clients[cid].linear_probing_training()
                    self.clients.local_evaluate(ood_eval=True, global_eval=True)


            # for stats_id, stats_item in enumerate(stats_list):
            #     print("Stats {} Evaluation".format(stats_id))
            #     stats = copy.deepcopy(stats_item)
            #     print("Stats: ", stats)
            #     id_tot_correct = stats[2].pop(stats_id)
            #     id_tot_auc = stats[3].pop(stats_id)
            #     id_num_samples = stats[1].pop(stats_id)

            #     id_tot_correct_list.append(id_tot_correct)
            #     id_tot_auc_list.append(id_tot_auc)
            #     id_num_samples_list.append(id_num_samples)

            #     ood_tot_correct_list.extend(stats[2])
            #     ood_tot_auc_list.extend(stats[3])
            #     ood_num_samples_list.extend(stats[1])

            # id_acc = sum(id_tot_correct_list)*1.0 / sum(id_num_samples_list)
            # id_auc = sum(id_tot_auc_list)*1.0 / sum(id_num_samples_list)

            # ood_acc = sum(ood_tot_correct_list)*1.0 / sum(ood_num_samples_list)
            # ood_auc = sum(ood_tot_auc_list)*1.0 / sum(ood_num_samples_list)

            # id_accs = [a / n for a, n in zip(id_tot_correct_list, id_num_samples_list)]
            # id_aucs = [a / n for a, n in zip(id_tot_auc_list, id_num_samples_list)]

            # ood_accs = [a / n for a, n in zip(ood_tot_correct_list, ood_num_samples_list)]
            # ood_aucs = [a / n for a, n in zip(ood_tot_auc_list, ood_num_samples_list)]

            # print("ID Performance (client model i on dataset i):")
            # print("ID Client-Num-Weighted Test Accurancy: {:.4f}".format(id_acc))
            # print("ID Client-Num-Weighted Test AUC: {:.4f}".format(id_auc))

            # print("ID Client-Equally Test Accurancy: {:.4f}".format(np.mean(id_accs)))
            # print("ID Client-Equally Test AUC: {:.4f}".format(np.mean(id_aucs)))

            # print("ID Std Test Accurancy: {:.4f}".format(np.std(id_accs)))
            # print("ID Std Test AUC: {:.4f}".format(np.std(id_aucs)))

            # print("OOD Performance (client model i on all other dataset j):")
            # print("OOD Client-Num-Weighted Test Accurancy: {:.4f}".format(ood_acc))
            # print("OOD Client-Num-Weighted Test AUC: {:.4f}".format(ood_auc))

            # print("OOD Client-Equally Test Accurancy: {:.4f}".format(np.mean(ood_accs)))
            # print("OOD Client-Equally Test AUC: {:.4f}".format(np.mean(ood_aucs)))

            # print("OOD Std Test Accurancy: {:.4f}".format(np.std(ood_accs)))
            # print("OOD Std Test AUC: {:.4f}".format(np.std(ood_aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def tta_eval(self):
        # compute sharpness
        # if self.monitor_hessian:
        #     print("Computing sharpness")
        #     eigenvals_list = []
        #     for client in self.select_clients():
        #         eigenvals, _ = client.compute_hessian_eigen()
        #         eigenvals_list.append(eigenvals[0])
        #     print("\nHessian eigenval list: ", eigenvals_list)
        #     print("\nHessian eigenval mean: ", np.mean(eigenvals_list))

        print("Testing time adaptation on OOD")
        ood_acc_list = []
        ood_auc_list = []
        client_ids = []
        for client in self.select_clients():
            client_ids.append(client.id)

        for client in self.select_clients():
            # loading trained model when tta evaluation only
            if self.test_time_adaptation_eval:
                # same parameter as save_item
                pt_dict = client.load_item(
                    item_name=self.goal,
                    item_path="models/" + self.dataset + "/",
                )
                client.model.load_state_dict(pt_dict.state_dict())

            print("Current client ID: ", client.id)
            copy_client_ids = copy.deepcopy(client_ids)
            print("copy_client_ids: ", copy_client_ids)
            copy_client_ids.remove(client.id)
            print("copy_client_ids removed: ", copy_client_ids)
            ood_acc, ood_auc = client.quick_adaptation_eval(dataset_ids=copy_client_ids)
            ood_acc_list.append(ood_acc)
            ood_auc_list.append(ood_auc)

        print("Average Test-Time Adaptation Accuracy: ", np.mean(ood_acc_list))
        print("Average Test-Time Adaptation AUC: ", np.mean(ood_auc_list))

    def server_pretrain(self):
        global_train_data = []
        fixed_seed = 43
        random.seed(fixed_seed)

        self.loss = nn.CrossEntropyLoss()
        if self.pretrained:
            self.optimizer = torch.optim.Adam(
                self.global_model.parameters(), lr=self.learning_rate
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.global_model.parameters(), lr=self.learning_rate
            )
            # self.optimizer = torch.optim.Adam(
            #     self.global_model.parameters(), lr=self.learning_rate
            # )

        for c_idx in range(self.num_clients):
            train_data = read_client_data(self.dataset, c_idx, is_train=True)
            sample_num = int(len(train_data) * self.server_global_dataset_rate)
            sampled_train_data = random.sample(train_data, sample_num)
            print("Sample {} from Clinet {}".format(sample_num, c_idx))
            global_train_data.extend(sampled_train_data)

        # noticing the batch_size should be small
        global_train_loader = DataLoader(
            global_train_data,
            int(self.batch_size * 0.25),
            drop_last=False,
            shuffle=True,
        )
        for step in range(self.global_steps):
            self.global_model.train()
            print("global loader len: ", len(global_train_loader))
            for i, (x, y) in enumerate(global_train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.global_model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                print("Training Loss: {:.4f}".format(loss / len(y)))
            print("Current Global Step: {}".format(step))
            self.global_model.eval()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.evaluate()
