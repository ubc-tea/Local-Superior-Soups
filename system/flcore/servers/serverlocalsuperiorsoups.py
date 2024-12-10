import time
from flcore.clients.clientlocalsuperiorsoups import clientLocalSuperiorSoups
from flcore.servers.serverbase import Server
import numpy as np


class LocalSuperiorSoups(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientLocalSuperiorSoups)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.server_training = args.server_training

    def train(self):

        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(ood_eval=False, global_eval=True)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        self.send_models()
        self.evaluate(ood_eval=False, global_eval=True)

        # ================= new function =================
        # compute sharpness
        if self.monitor_hessian:
            print("Computing sharpness")
            eigenvals_list = []
            for client in self.select_clients():
                eigenvals, _ = client.compute_hessian_eigen()
                eigenvals_list.append(eigenvals[0])
            print("\nHessian eigenval list: ", eigenvals_list)
            print("\nHessian eigenval mean: ", np.mean(eigenvals_list))

        # testing time adaptation
        if self.test_time_adaptation:
            self.tta_eval()

        # save each client model item
        for client in self.select_clients():
            client.save_item(
                item=client.model,
                item_name=self.goal,
                item_path="models/" + self.dataset + "/",
            )

        self.save_results()
        self.save_global_model()
