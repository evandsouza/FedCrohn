import flwr as fl
import numpy as np
import sources.GraphConv as GCN
import torch as t
from collections import OrderedDict

# small helper functions

def get_params(model: t.nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: t.nn.Module, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: t.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class SyntheticClient(fl.client.NumPyClient):
    def __init__(self, cid, X, Y, gene_list=None, env_dim=4):
        self.cid = cid
        self.X = X
        self.Y = Y
        self.env_dim = env_dim
        # instantiate model
        gene_size = X[0].shape[1]
        num_genes = X[0].shape[0]
        self.model = GCN.BaselineNN(gene_size, num_genes, None, gene_list or [], name=f"client_{cid}", env_dim=env_dim)
        self.wrapper = GCN.NNwrapper(self.model)
        # attach synthetic envs
        self.wrapper.E = np.random.rand(len(X), env_dim).astype(np.float32)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        # short local training
        self.wrapper.fit(self.X, self.Y, epochs=2, batch_size=4, save_model_every=1000, warmStart=0, weight_decay=0.0, learning_rate=1e-3, silent=True)
        return get_params(self.model), len(self.X), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        preds = self.wrapper.predict(self.X, batch_size=len(self.X))
        preds_bin = [1 if p>0.5 else 0 for p in preds]
        acc = float(sum(int(a==b) for a,b in zip(preds_bin, self.Y))/len(self.Y))
        # return loss, num_examples, metrics
        return 1.0-acc, len(self.X), {"accuracy": acc}


if __name__ == "__main__":
    # create synthetic dataset per client
    num_clients = 3
    num_samples = 12
    num_genes = 12
    gene_size = 5
    env_dim = 4

    datasets = []
    for c in range(num_clients):
        X = [np.random.randn(num_genes, gene_size).astype(np.float32) for _ in range(num_samples)]
        Y = np.random.randint(0,2,size=(num_samples,)).tolist()
        datasets.append((X, Y))

    def client_fn(cid: str):
        idx = int(cid)
        X, Y = datasets[idx]
        return SyntheticClient(cid, X, Y, gene_list=None, env_dim=env_dim)

    # instantiate a global model for evaluation
    global_model = GCN.BaselineNN(gene_size, num_genes, None, [], name="global", env_dim=env_dim)

    strategy = fl.server.strategy.FedAvg(evaluate_fn=None)
    res = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )

    print("Simulation finished, metrics (centralized):", getattr(res, 'metrics_centralized', None))
