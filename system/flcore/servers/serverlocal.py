from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()