from typing import List, Tuple
from flwr.common import Metrics
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
	TEST_DS = args[1]
	DB = "marshalledP3/cagi"+TEST_DS+".multianno.missenseFalse.RegionsNone.m.min"+str(MIN_REFERENCE_NUM)
	weightPhenoPGenes, phenoPGenes = readPhenopedia("phenopediaCrohnGenes/CrohnGenes.txt")	
	geneList = sorted(pickle.load(open("marshalledP3/totGeneSet.m.min"+str(MIN_REFERENCE_NUM), "rb")))
	strategy = fl.server.strategy.FedAvg()
	#read data, process them
	###########################
	datasets = {}
	HX = {} # {sample:(vector, label)}
	print("Preparing %s"%(TEST_DS))
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

	net = GCN.BaselineNN(len(X[0][0]), len(X[0]), None, geneList, "baseline_")
# Define strategy
	#strategy = fl.server.strategy.FedAdagrad(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(net, X, Y), initial_parameters=fl.common.ndarrays_to_parameters(get_params(net)), eta=1e-2, eta_l=1e-3)	#strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(net, X, Y))#, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
	#strategy = fl.server.strategy.FedAdam(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(net, X, Y), initial_parameters=fl.common.ndarrays_to_parameters(get_params(net)), eta=1e-2, eta_l=1e-3)	#strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(net, X, Y))#, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
	strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(net, X, Y))#, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)

# Start Flower server
	fl.server.start_server(
		server_address="0.0.0.0:8080",
		config=MyConfig,
		strategy=strategy,
	)

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
		return mcc, {"auc": float(aucScoreGood), "auprc":auprc}
	return evaluate
	
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
	# Multiply accuracy of each client by number of examples used
	mauc = [num_examples * m["auc"] for num_examples, m in metrics]
	mauprc = [num_examples * m["auprc"] for num_examples, m in metrics]
	examples = [num_examples for num_examples, _ in metrics]

	# Aggregate and return custom metric (weighted average)
	return {"auc": sum(mauc) / sum(examples), "auprc": sum(mauprc) / sum(examples)}

def get_params(model: t.nn.ModuleList) -> List[np.ndarray]:
	"""Get model weights as a list of NumPy ndarrays."""
	return [val.cpu().numpy() for _, val in model.state_dict().items()]

class MyConfig(fl.server.ServerConfig):
	num_rounds = 5
	round_timeout = None



if __name__ == "__main__":
	main(sys.argv)


