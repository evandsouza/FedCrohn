#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  analysisVCF.py
#  
#  Copyright 2017 Daniele Raimondi <eddiewrc@mira>
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
import pickle
import sys, os, time, random, copy, math
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


class StructuredScaler:
	def __init__(self, scaler):
		self.scaler = scaler
		
	def fit_transform(self, X):
		if type(X) != np.array:
			X = np.array(X)
		shape = X.shape
		#print shape
		#raw
		X = X.reshape(shape[0]*shape[1], shape[2])
		X = self.scaler.fit_transform(X)
		return X.reshape(shape[0], shape[1], shape[2])
		
	def transform(self, X):
		if type(X) != np.array:
			X = np.array(X)
		shape = X.shape
		X = X.reshape(shape[0]*shape[1], shape[2])
		X = self.scaler.transform(X)
		return X.reshape(shape[0], shape[1], shape[2])
	
def shuffleColumns(X):
	X = np.array(X)	
	Xs = np.zeros(X.shape)
	i = 0
	while i < X.shape[1]:
		np.random.shuffle(X[:,i])
		Xs[:,i] = X[:,i]
		i += 1
	#print Xs[:,0], X[:,0]
	assert Xs.shape == X.shape
	return Xs
	 
def checkVectors(a,b):
	assert b.count(b[0]) != len(b)
	assert len(a) == len(b)
	assert len(a) > 0
	l = len(a[0])
	for e in a:
		assert len(e) == l
		for g in e:
			for i in g:
				#print i	
				if math.isnan(i):				
					raise Exception("NAN found.")
	print ("Vectors with %d dimensions." % l)
	return len(a), l
	
def buildFeatVect(HX, samples): # {sample:(vector, label)}
	X = []
	Y = []	
	for s in samples:		
		Y.append(HX[s][1])		
		X.append(HX[s][0])
	assert len(X) == len(Y)
	return X, Y


def buildVectorGeneWise(genedb, geneList, weightPhenoPGenes, gwasHist):	
	vect = []
	totWG = sum(weightPhenoPGenes.values())
	for k in geneList:      #0    1     2       3      4       5    6   7    8    9         10    11
		varList = genedb[k]#pos, vest, region, gene, vartype, rec, hi, rvis, gdi, metasvm, mcap, crom
		hist =  countVars(varList)		
		if len(varList) < 1:
			#vect.append(([0]*NUMBINS + [0,0])*2 + [0,0,0,0]+[0]+[0]*10)
			vect.append([0]*len(CONST.TYPES)+[0,0])
			continue
		#tmp = []+hist.values()+	[weightPhenoPGenes[k]/float(totWG)]#, getHistCount(varList[0][11], varList[0][0], gwasHist)]
			
		tmp = []+CONST.getOrderedValues(hist)+	[weightPhenoPGenes[k]/float(totWG), varList[0][7]]#+getVarEffPreds(varList,9)+getVarEffPreds(varList,10)
		if SHUFFLE:
			random.shuffle(tmp)
		vect.append(tmp)
	return vect

def countVars(varlist):
	types = copy.deepcopy(CONST.TYPES)
	for v in varlist: #pos, vest, region, gene, vartype, rec, hi, rvis, gdi, metasvm, mcap
		if "exonic" == v[2] and "synonymous SNV" == v[4]:
			continue
		if "intergenic" == v[2]:
			raiseException("Intergenic should not be present!")#	continue
		types[v[2]] +=1
	return types

def getGeneScores(vl):
	if len(vl) > 0:
		return [vl[0][10], vl[0][11], vl[0][12]]
	else:
		return [0,0,0]
			
def getVarEffPreds(vl, pos):
	scores = []
	for v in vl:
		if v[pos] != None:
			scores.append(v[pos])
	hist = (np.histogram(scores, NUMBINS, [0,1])[0]/ float(len(scores))).tolist()
	assert abs(sum(hist) - 1) < 0.00001
	if len(scores) > 0:
		return hist + [np.mean(scores), max(scores)]		
	else:
		return [0]*NUMBINS + [0,0]	
				
def scanGenes(exome, geneList): #extracts gene level info without processin it too much
	geneDB = {}
	for g in sorted(geneList):
		geneDB[g] = []
	for crom in exome:			
		cromName = crom[0]
		varList = crom[1]			
		for var in varList:	
			#print var[3]
			if var[3] not in geneList:
				raise Exception(" >> ERROR: gene %s not found" % var[3])
				continue

			geneDB[var[3]].append(list(var)+[crom2int(crom[0][3:])])
			
	#for i in geneDB.items():
	#	print i
	#raw_input()		
	return geneDB #list of variants foreach gene

def crom2int(s):
	try:
		return int(s)
	except:
		pass
	if s == "Y":
		return 23
	elif s == "X":
		return 24
	raise Exception("%s is not a valid cromosome name" % s)


if __name__ == '__main__':
	import sys
	sys.exit(main())
