import sys, os, time, random, copy, math, pickle
from sklearn.model_selection import StratifiedKFold, KFold
from flwr.common import Metrics
from sources.iongreen2_analysisPaper import scanGenes, buildFeatVect, buildVectorGeneWise, checkVectors
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
CAGI_TEST = 3
import flwr as fl
from flwr.common.typing import Scalar
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List
import torch as t
DBs = ["marshalledP3/cagi2.multianno.missenseFalse.RegionsNone.m.min"+str(MIN_REFERENCE_NUM),"marshalledP3/cagi3.multianno.missenseFalse.RegionsNone.m.min"+str(MIN_REFERENCE_NUM),"marshalledP3/cagi4.multianno.missenseFalse.RegionsNone.m.min"+str(MIN_REFERENCE_NUM)]

def main(args):

	CV_FOLDS = int(args[1])
	NUM_CLIENTS = CV_FOLDS -1

	weightPhenoPGenes, phenoPGenes = readPhenopedia("phenopediaCrohnGenes/CrohnGenes.txt")	
	geneList = sorted(pickle.load(open("marshalledP3/totGeneSet.m.min"+str(MIN_REFERENCE_NUM), "rb")))
	#read data, process them
	###########################
	DATAX = []
	DATAY = []
	HX = {} # {sample:(vector, label)}
	code = 2
	for D in DBs:
		print("Preparing %s, %d"%(D, code))
		db = pickle.load(open(D, "rb")) #{db = {CAGI_ID: (annovarAnnotations, LABEL}}
		print ("Read %d exomes from %s" % (len(db), D))
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
		assert len(Y) == len(train) == len(X)
		print ("Train size: %d" % len(train))
		print ("Length vectors: ", len(X[0]), len(X[0][0]))
		
		shapeX = checkVectors(X,Y)
		code += 1
		DATAX += X
		DATAY += Y
		#random.shuffle(DATAY)
	print(len(DATAX), len(DATAY))
	datasets = splitData(DATAX, DATAY, CV_FOLDS)
	#datasets = splitData(DATAX, DATAY, CV_FOLDS)

	totRes = {"sen":[], "spe":[], "pre":[], "mcc":[], "auc":[], "auprc":[]}
	i = 0
	while i < CV_FOLDS:
		print("Start fold %d"%i)
		tmpTest = datasets.pop(0)
		assert len(datasets) == CV_FOLDS -1
		def client_fn(cid: str):
			assert len(datasets) == CV_FOLDS -1
			# create a single client instance
			print(cid)
			return CagiRealClient(cid, datasets, geneList)

		testX = tmpTest[0]
		testY = tmpTest[1]
		net = GCN.BaselineNN(len(testX[0][0]), len(testX[0]), None, geneList, "baseline_")
		strategy = fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn(net, testX, testY), fraction_fit=1, fraction_evaluate=1, min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS)#, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
		# start simulation
		tmpRes = fl.simulation.start_simulation(
			client_fn=client_fn,
			clients_ids = range(0,NUM_CLIENTS),
			num_clients=NUM_CLIENTS,
			config=fl.server.ServerConfig(num_rounds=5),
			strategy=strategy)
		datasets.append(tmpTest)
		print(tmpRes.metrics_centralized)
		totRes["sen"].append(tmpRes.metrics_centralized["sen"][-1][1])
		totRes["spe"].append(tmpRes.metrics_centralized["spe"][-1][1])
		totRes["pre"].append(tmpRes.metrics_centralized["pre"][-1][1])
		totRes["mcc"].append(tmpRes.metrics_centralized["mcc"][-1][1])
		totRes["auc"].append(tmpRes.metrics_centralized["auc"][-1][1])
		totRes["auprc"].append(tmpRes.metrics_centralized["auprc"][-1][1])
		print("Fold finished")
		#input()
		del client_fn, net, strategy
		i+=1

	print("Final results:")
	for r in totRes.items():
		print(r[0], np.mean(r[1]), np.std(r[1]))

# Flower client, adapted from Pytorch quickstart example
class CagiRealClient(fl.client.NumPyClient):
	def __init__(self, cid, data, geneList):
		print("Creating client ", str(cid))
		self.X = data[int(cid)][0]
		self.Y = data[int(cid)][1]	
		self.cid = cid
		# Instantiate model
		self.net = GCN.BaselineNN(len(self.X[0][0]), len(self.X[0]), None, geneList, "baseline_")
		self.wrapper = GCN.NNwrapper(self.net)

	def get_parameters(self, config):
		return get_params(self.net)

	def fit(self, parameters, config):
		set_params(self.net, parameters)
		self.wrapper.fit(self.X, self.Y, epochs=100, batch_size=3, weight_decay = 1, learning_rate = 1e-3, silent=True, save_model_every=1000)
		# Return local model and statistics
		return get_params(self.net), len(self.X), {}

	def evaluate(self, parameters, config):
		set_params(self.net, parameters)
		Yp = self.wrapper.predict(self.X, batch_size=len(self.X))

		sen, spe, acc, bac, pre, mcc, aucScoreGood, auprc = U.getScoresSVR(Yp, self.Y, threshold=None, invert = False, PRINT = False, CURVES = False, SAVEFIG=None)	
		# Return statistics
		return mcc, len(self.X), {"auc": float(aucScoreGood), "auprc":auprc}

def get_params(model: t.nn.ModuleList) -> List[np.ndarray]:
	"""Get model weights as a list of NumPy ndarrays."""
	return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: t.nn.ModuleList, params: List[np.ndarray]):
	"""Set model weights from a list of NumPy ndarrays."""
	params_dict = zip(model.state_dict().keys(), params)
	state_dict = OrderedDict({k: t.from_numpy(np.copy(v)) for k, v in params_dict})
	model.load_state_dict(state_dict, strict=True)# Start simulation (a _default server_ will be created)

def splitData(x, y, folds, shuffle=True):
	d = []
	assert len(x) == len(y)
	for i, _ in enumerate(x):
		d.append((x[i], y[i]))
	if shuffle:
		random.shuffle(d)
	X = []
	Y = []
	for i, _ in enumerate(d):
		X.append(d[i][0])
		Y.append(d[i][1])
	X = np.array(X)
	Y = np.array(Y)
	#skf = KFold(n_splits=folds, shuffle=shuffle)
	skf = StratifiedKFold(n_splits=folds, shuffle=shuffle)
	datasets = []
	for train, test in skf.split(X, Y):
		datasets.append((list(X[test]), list(Y[test])))
	
	s = 0
	for ds in datasets:
		print(len(ds[0]), sum(ds[1]))
		s+=len(ds[0])
		assert len(ds[0]) == len(ds[1])
	assert len(x) == s
	return datasets


def splitDataMine(x, y, folds, shuffle=True):
	d = []
	assert len(x) == len(y)
	for i, _ in enumerate(x):
		d.append((x[i], y[i]))
	if shuffle:
		random.shuffle(d)
	l = math.floor(len(x)/float(folds))
	datasets = {}
	f = 0
	while f < folds:
		datasets[f] = [[],[]]
		start = f*l
		while start < (f*l)+l:
			datasets[f][0].append(x[start])
			datasets[f][1].append(y[start])
			start += 1
		if f == folds-1:
			while start < len(x):
				datasets[f][0].append(x[start])
				datasets[f][1].append(y[start])
				start += 1
		f+=1
	s = 0
	for ds in datasets.values():
		print(len(ds[0]))
		s+=len(ds[0])
		assert len(ds[0]) == len(ds[1])
	assert len(x) == s
	return list(datasets.values())

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
	# Multiply accuracy of each client by number of examples used
	mauc = [num_examples * m["auc"] for num_examples, m in metrics]
	mauprc = [num_examples * m["auprc"] for num_examples, m in metrics]
	examples = [num_examples for num_examples, _ in metrics]

	# Aggregate and return custom metric (weighted average)
	return {"auc": sum(mauc) / sum(examples), "auprc": sum(mauprc) / sum(examples)}


def get_evaluate_fn(model, X, Y):
	"""Return an evaluation function for server-side evaluation."""
	# The `evaluate` function will be called after every round
	def evaluate(server_round, parameters, config):
		print("Server evaluation")
		params_dict = zip(model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: t.tensor(v) for k, v in params_dict})
		model.load_state_dict(state_dict, strict=True)
		wrapper = GCN.NNwrapper(model)
		Yp = wrapper.predict(X, batch_size=len(X))

		sen, spe, acc, bac, pre, mcc, aucScoreGood, auprc = U.getScoresSVR(Yp, Y, threshold=None, invert = False, PRINT = True, CURVES = False, SAVEFIG=None)	
		# Return statistics
		return mcc, {"sen":sen, "spe":spe, "pre":pre, "mcc":mcc, "auc":aucScoreGood, "auprc":auprc}
	return evaluate
	


if __name__ == "__main__":
	main(sys.argv)


