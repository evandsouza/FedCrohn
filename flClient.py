import sys, os, time, random, copy, math, pickle
from sources.iongreen2_analysisPaper import scanGenes,  buildFeatVect, buildVectorGeneWise, checkVectors
import numpy as np
from sources import parseAnnovarMultianno as PAM
from sources.readPhenopedia import readPhenopedia
from sources import utils as U
import sources.GraphConv as GCN
from sklearn.preprocessing import StandardScaler
import sources.Constants as CONST
SCRATCH = False
CHECK_COUNTS = False
PATHWAY_NEIGHBORS = False
MIN_REFERENCE_NUM = 0 
SCALER = False
SHUFFLE = False
NUMBINS = 3
STORE_ACTIVATIONS = True 
CAGI_TRAIN = 4
CAGI_TEST = 3
import flwr as fl
from flwr.common.typing import Scalar
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List
import torch as t
numCenters = 2

def main(args):
	TRAIN_DS = args[1]
	DB = "marshalledP3/cagi"+TRAIN_DS+".multianno.missenseFalse.RegionsNone.m.min"+str(MIN_REFERENCE_NUM)
	weightPhenoPGenes, phenoPGenes = readPhenopedia("phenopediaCrohnGenes/CrohnGenes.txt")	
	geneList = sorted(pickle.load(open("marshalledP3/totGeneSet.m.min"+str(MIN_REFERENCE_NUM), "rb")))
	strategy = fl.server.strategy.FedAvg()
	#read data, process them
	###########################
	datasets = {}
	HX = {} # {sample:(vector, label)}
	print("Preparing %s"%(TRAIN_DS))
	db = pickle.load(open(DB, "rb")) #{db = {CAGI_ID: (annovarAnnotations, LABEL}}
	print ("Read %d exomes from %s" % (len(db), DB))
	for h in db.items():
		sampleName = h[0]
		exome = h[1][0]
		label = h[1][1]
		geneDB = scanGenes(exome.items(), geneList) #extracts gene-wise data without processing it		
		HX[sampleName] = (buildVectorGeneWise(geneDB, geneList, weightPhenoPGenes, None), label)
	train = HX.keys()

	X, Y = buildFeatVect(HX, train)
	count = np.zeros(len(CONST.TYPES))
	if CHECK_COUNTS:
		for s in X:
			for g in s:
				count += np.array(g[:-2])
				#print g
	scaler = None
	if SCALER:
		scaler = StructuredScaler(StandardScaler())
		X = scaler.fit_transform(X)	
	print (scaler)
	assert len(Y) == len(train) == len(X)
	print ("Train size: %d" % len(train))
	print ("Length vectors: ", len(X[0]), len(X[0][0]))
	
	shapeX = checkVectors(X,Y)
	DATA = (X, Y)	
	fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CagiRealClient(DATA, geneList))


# Flower client, adapted from Pytorch quickstart example
class CagiRealClient(fl.client.NumPyClient):
	def __init__(self, data, geneList):
		print("Creating client ")
		self.X = data[0]
		self.Y = data[1]	
		# Instantiate model
		self.net = GCN.BaselineNN(len(self.X[0][0]), len(self.X[0]), None, geneList, "baseline_")
		self.wrapper = GCN.NNwrapper(self.net)

	def get_parameters(self, config):
		return get_params(self.net)

	def fit(self, parameters, config):
		set_params(self.net, parameters)
		self.wrapper.fit(self.X, self.Y, epochs=100, batch_size=3, weight_decay = 1, learning_rate = 1e-3)
		# Return local model and statistics
		return get_params(self.net), len(self.X), {}

	def evaluate(self, parameters, config):
		set_params(self.net, parameters)
		Yp = self.wrapper.predict(self.X, batch_size=len(self.X))

		sen, spe, acc, bac, pre, mcc, aucScoreGood, auprc = U.getScoresSVR(Yp, self.Y, threshold=None, invert = False, PRINT = True, CURVES = False, SAVEFIG=None)	
		# Return statistics
		return mcc, len(self.X), {"auc": float(aucScoreGood), "auprc":auprc}

def get_params(model: t.nn.ModuleList) -> List[np.ndarray]:
	"""Get model weights as a list of NumPy ndarrays."""
	return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: t.nn.ModuleList, params: List[np.ndarray]):
	"""Set model weights from a list of NumPy ndarrays."""
	params_dict = zip(model.state_dict().keys(), params)
	state_dict = OrderedDict({k: t.from_numpy(np.copy(v)) for k, v in params_dict})
	model.load_state_dict(state_dict, strict=True)


'''def get_evaluate_fn(testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
	"""Return an evaluation function for centralized evaluation."""

	def evaluate(
		server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
	) -> Optional[Tuple[float, float]]:
		"""Use the entire CIFAR-10 test set for evaluation."""

		# determine device
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		model = Net()
		set_params(model, parameters)
		model.to(device)

		testloader = torch.utils.data.DataLoader(testset, batch_size=50)
		loss, accuracy = test(model, testloader, device=device)

		# return statistics
		return loss, {"accuracy": accuracy}

	return evaluate
'''

# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#	clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#	Also, the global model is evaluated on the valset partition residing in each
#	client. This is useful to get a sense on how well the global model can generalise
#	to each client's data.
if __name__ == "__main__":
	main(sys.argv)


