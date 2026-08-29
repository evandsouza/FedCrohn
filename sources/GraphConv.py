#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pred1.py
#  
#  Copyright 2018 Daniele Raimondi <eddiewrc@vega>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import matplotlib.pyplot as plt
import os, copy, random, time, math
from sys import stdout
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
#t.manual_seed(0)

class DenseNN(t.nn.Module):

	def __init__(self, size, p, geneList, name = "NN"):
		super(DenseNN, self).__init__()		
		os.system("mkdir -p models")
		self.name = name
		
		self.size = size		
		self.nn = t.nn.Sequential( t.nn.Linear(self.size, 100), t.nn.LeakyReLU(), t.nn.Dropout(0.1), t.nn.Linear(100, 1))
		#self.final = t.nn.Sequential(t.nn.Dropout(0.1), t.nn.Linear(self.numGenes, 10), t.nn.LeakyReLU(), t.nn.Linear(10,1))
		self.apply(self.init_weights)
		
	def forward(self, x, GET_ACT = False):	
		#print x.size()	
		o = self.nn(x)
		return o
	
	def init_weights(self, m):
		if isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Bilinear) or isinstance(m, GraphConvolution):
			print ("Initializing weights...", m.__class__.__name__)
			t.nn.init.normal(m.weight, 0, 0.01)
			#t.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(1)
		elif isinstance(m, t.nn.Embedding):
			print ("Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform(m.weight)		
		

class GraphConvolution(t.nn.Module):
	
	def __init__(self, in_features, out_features, adjMatrix, bias=True):
		super(GraphConvolution, self).__init__()
		self.adj = adjMatrix
		self.in_features = in_features
		self.out_features = out_features
		self.weight = t.nn.Parameter(t.FloatTensor(in_features, out_features))
		if bias:
			self.bias = t.nn.Parameter(t.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		#self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		support = t.matmul(input.squeeze(), self.weight)
		output = t.matmul(self.adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

class BaselineNN(t.nn.Module):
	def __init__(self, genesize, numGenes, p, geneList, name = "NN", env_dim=0):
		super(BaselineNN, self).__init__()		
		os.system("mkdir -p models")
		self.name = name
		
		self.geneInputSize = genesize
		self.numGenes = numGenes
		self.env_dim = env_dim
		
		self.nn = t.nn.Sequential( t.nn.Linear(self.geneInputSize, 1), t.nn.LeakyReLU())
		final_in = self.numGenes
		if self.env_dim and self.env_dim > 0:
			# create small env encoder
			self.env_encoder = t.nn.Sequential(t.nn.Linear(self.env_dim, 16), t.nn.ReLU(), t.nn.Linear(16, 8), t.nn.ReLU())
			final_in = self.numGenes + 8
		else:
			self.env_encoder = None

		self.final = t.nn.Sequential(t.nn.Dropout(0.1), t.nn.Linear(final_in, 1))
		#self.final = t.nn.Sequential(t.nn.Dropout(0.1), t.nn.Linear(self.numGenes, 10), t.nn.LeakyReLU(), t.nn.Linear(10,1))
		self.apply(self.init_weights)
		
	def forward(self, x, env=None, GET_ACT = False):    
		# x: [batch, num_genes, geneInputSize]
		o1 = self.nn(x)  # [batch, num_genes, 1]
		gene_feat = o1.squeeze(2) if o1.dim() == 3 else o1.squeeze()
		# gene_feat: [batch, num_genes]
		if self.env_encoder is not None:
			if env is None:
				raise ValueError("Environment tensor is required when env_dim > 0.")
			env_emb = self.env_encoder(env) * 3.0
			combined = t.cat([gene_feat, env_emb], dim=1)
		else:
			combined = gene_feat

		out = self.final(combined)
		if GET_ACT:
			return gene_feat, out
		else:
			return out

	def get_gene_importance(self):
		"""Return absolute contribution of each gene inferred from final layer weights."""
		# final linear is self.final[1]
		try:
			lin = self.final[1]
			w = lin.weight.data.abs().cpu().numpy()  # shape (1, final_in)
			# first self.numGenes entries correspond to gene contributions
			gene_w = w[0, :self.numGenes]
			# normalize
			if gene_w.sum() == 0:
				return gene_w
			return gene_w / (gene_w.max() + 1e-12)
		except Exception:
			return None

	def get_env_importance(self):
		"""Return absolute mean importance of each environment feature if encoder present."""
		if getattr(self, 'env_encoder', None) is None:
			return None
		try:
			first_layer = self.env_encoder[0]
			w = first_layer.weight.data.abs().cpu().numpy()  # shape (out, in)
			imp = w.mean(axis=0)
			if imp.sum() == 0:
				return imp
			return imp / (imp.max() + 1e-12)
		except Exception:
			return None
	
	def init_weights(self, m):
		if isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Bilinear) or isinstance(m, GraphConvolution):
			print ("Initializing weights...", m.__class__.__name__)
			t.nn.init.normal(m.weight, 0, 0.01)
			#t.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(1)
		elif isinstance(m, t.nn.Embedding):
			print( "Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform(m.weight)		
			
class GraphConvModel(t.nn.Module):
	
	def __init__(self, genesize, numGenes, p, geneList, name = "NN"):
		super(GraphConvModel, self).__init__()		
		os.system("mkdir -p models")
		self.name = name
		##########DEFINE NN HERE##############
		#self.adj = Variable(self.buildAdj(p, geneList))
		#print self.adj.size()
		#self.buildDegreeMatrix(self.adj.data)
		self.adj = Variable(self.buildSymmNormAdj(p, geneList))
		print (self.adj.size())
		self.geneInputSize = genesize
		self.numGenes = numGenes
		
		self.gcn = t.nn.Sequential(t.nn.Dropout(0.2), GraphConvolution(self.geneInputSize, 5, self.adj), t.nn.Dropout(0.4), t.nn.LeakyReLU(),\
		GraphConvolution(5, 5, self.adj), t.nn.Dropout(0.4), t.nn.LeakyReLU(),\
		GraphConvolution(5, 5, self.adj), t.nn.Dropout(0.4), t.nn.LeakyReLU(),\
		
		GraphConvolution(5, 1, self.adj), t.nn.Dropout(0.4), t.nn.LeakyReLU())
		self.final = t.nn.Sequential(t.nn.Linear(self.numGenes, 1))
		self.apply(self.init_weights)
		#raw_input()
		'''
		newbest
		Sen = 0.692
		Spe = 0.617
		Acc = 0.661 
		Bac = 0.655
		Pre = 0.714
		MCC = 0.308
		#AUC = 0.659
		#AUPRC= 0.725

		'''
	
	def init_weights(self, m):
		if isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Bilinear) or isinstance(m, GraphConvolution):
			print( "Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(0.01)
		elif isinstance(m, t.nn.Embedding):
			print( "Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform(m.weight)
			
	def forward(self, x):	
		#print x.size()	
		o = self.gcn(x)
		#print o.size()
		o = self.final(o.squeeze())
		#o = (t.sum(o.squeeze(), 1)/o.size(1)).unsqueeze(1)		
		return o

	def buildSymmNormAdj(self, p, geneList):
		adj = self.buildAdj(p, geneList)
		d = self.buildDegreeMatrix(adj)
		tmp = t.matmul(t.matmul(d, adj), d)
		return tmp

	def buildDegreeMatrix(self, adj):
		tmp = t.sum(adj,0)
		tmp = t.pow(tmp, -0.5)
		#raw_input()
		tmp = t.diag(tmp)
		#print tmp.size()
		#raw_input()
		return tmp

	def buildAdj(self, p, geneList):	
		print ("Preparing path adj")
		mask = t.zeros(len(geneList), len(geneList))
		padding = []
		i = 0
		while i < len(geneList):
			#print "Working on gene: %d/%d" % (i, len(geneList))
			tmp = self.getNeighbors(i, geneList, p)
			for g in tmp:
				mask[i][g] = 1.0
			i+=1
		mask  = mask + t.eye(len(geneList), len(geneList))	
		print ("Done adj.")
		return mask

	def getNeighbors(self, genePos, geneList, path):	
		g = geneList[genePos]	
		neighbors = path.sharePath(g)		
		added = []
		for i in geneList:
			if i in neighbors:
				added.append(i)	
		if len(added) == 0:
			added.append(g)		
		assert len(added) == len(set(added))			
		nadded = []
		for g in added:
			nadded.append(geneList.index(g)) #vect containing list of gene positions in the vectors	
		return nadded

class myDataset(Dataset):
    
	def __init__(self, X, Y, E=None):    
		if Y == None:
			Y = []
			for x in X:
				Y.append([0]*len(x))        
		assert len(Y) == len(X)        
		self.X = t.FloatTensor(X)
		self.Y = t.LongTensor(Y)
		if E is not None:
			self.E = t.FloatTensor(E)
		else:
			self.E = None

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if self.E is None:
			return self.X[idx], self.Y[idx]
		else:
			return self.X[idx], self.Y[idx], self.E[idx]

class NNwrapper():

	def __init__(self, model):
		if type(model) == str:
			self.model = t.load(model)
		else:
			self.model = model
    
	def fit(self, originalX, originalY, epochs = 50, batch_size=20, save_model_every=10, warmStart = 0, weight_decay = 0.001, learning_rate = 1e-3, silent = False):
		########DATASET###########
		env_data = getattr(self, 'E', None)
		dataset = myDataset(originalX, originalY, env_data)
        
		#######MODEL##############        
        
		self.model.train()    
		print ("Start training")
		########LOSS FUNCTION######
		lossfn = LossWrapperCE(t.nn.CrossEntropyLoss(weight=None, size_average=False, ignore_index=-1, reduce=True), dummyColumn=True)
        
		########OPTIMIZER##########    
		self.learning_rate = learning_rate        
		parameters = self.model.parameters()
		optimizer = t.optim.RMSprop(parameters, lr=self.learning_rate, weight_decay=weight_decay)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        
		########DATALOADER#########
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
        
		e = 1 + warmStart
		minLoss = 1000000000
		ofp = open("models/"+self.model.name+".TrainLog","w")        
		while e < epochs  + warmStart:            
			errTot = 0
			i = 1
			start = time.time()
			for sample in loader:
				optimizer.zero_grad()
				# sample can be (x,y) or (x,y,e)
				if len(sample) == 2:
					x, y = sample
					yp = self.model.forward(Variable(x))
				else:
					x, y, evec = sample
					yp = self.model.forward(Variable(x), Variable(evec))
				if len(yp.shape) == 1:
					continue
				loss = lossfn(yp, Variable(y))
				loss.backward()    
				optimizer.step()
				errTot += loss.data
				i+=batch_size                        
			end = time.time()        
			print (" epoch %d, ERRORTOT: %f (%fs)" % (e, errTot, end-start))
			scheduler.step(errTot)
			if e % save_model_every == 0:
				print ("Store model ", e)
				t.save(self.model, "models/"+self.model.name+".iter_"+str(e)+".t")                
			stdout.flush()    
			e += 1    
		t.save(self.model, "models/"+self.model.name+".final.t")
		ofp.close()        
    
	def predict(self, X, Y = None, batch_size=-1, GET_ACT = False):
		self.model.eval()
		if batch_size == -1:
			batch_size = len(X)
		print ("Predicting...")
		env_data = getattr(self, 'E', None)
		dataset = myDataset(X, Y, env_data)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
		preds1 = []
		act = []
		for sample in loader:
			if len(sample) == 2:
				x, y = sample
				if GET_ACT == False:
					y_pred = t.nn.functional.sigmoid(self.model.forward(Variable(x)))
					preds1 += y_pred.data.squeeze().tolist()
				else:
					gene_act, y_pred = self.model.forward(Variable(x), GET_ACT = GET_ACT)
					y_pred = t.nn.functional.sigmoid(y_pred)
					preds1 += y_pred.data.squeeze().tolist()
					act += gene_act.data.squeeze().tolist()
			else:
				x, y, evec = sample
				if GET_ACT == False:
					y_pred = t.nn.functional.sigmoid(self.model.forward(Variable(x), Variable(evec)))
					preds1 += y_pred.data.squeeze().tolist()
				else:
					gene_act, y_pred = self.model.forward(Variable(x), Variable(evec), GET_ACT = GET_ACT)
					y_pred = t.nn.functional.sigmoid(y_pred)
					preds1 += y_pred.data.squeeze().tolist()
					act += gene_act.data.squeeze().tolist()
		if GET_ACT:
			return act, preds1
		else:
			return preds1

		
	

class LossWrapperCE():
	
	def __init__(self, loss,  dummyColumn=None):
		self.loss = loss
		self.dummyColumn = dummyColumn
		
	def __call__(self, input, target):
		if self.dummyColumn == None:
			input = t.cat([1-input, input],1)
		else:
			tmp = Variable(t.ones(input.size()[0], input.size()[1]))
			if input.is_cuda:
				tmp = tmp.cuda()		
			input = t.cat([tmp, input],1)
		return self.loss(input, target)
	
if __name__ == '__main__':
	from iongreen2 import main
	main()
	#mainEVAL()
