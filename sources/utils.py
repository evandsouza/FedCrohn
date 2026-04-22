#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  utils.py
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
import scipy.stats, random, math, time
import numpy as np


class Zscore:
	def __init__(self, v):
		self.mean = np.mean(v)
		self.std = np.std(v)
		self.distro = scipy.stats.norm(loc=0, scale=1)
		print( "Mean: ", self.mean, " Std: ", self.std)
	
	def zscore(self, x):
		return (x-self.mean)/self.std
		
	def pval(self, x):
		return 1-self.distro.cdf(x)	

def shuffleColumns(X):
	X = np.array(X)	
	Xs = np.zeros(X.shape)	
	i = 0
	while i < X.shape[1]:
		random.seed(time.time())		
		Xs[:,i] = random.sample(X[:,i], X.shape[0])
		i += 1
	#print Xs[:,0], X[:,0]
	assert Xs.shape == X.shape
	return Xs
	 
def checkVectors(a,b):
	assert b.count(b[0]) != len(b)
	assert len(a) == len(b)
	assert len(a) > 0
	l = len(a[0])
	for i in a:
		assert len(i) == l
		for j in i:
			#print i	
			if math.isnan(j):				
				raise Exception("NAN found.")
	print ("Vectors with %d dimensions." % l)
	return len(a), l
	
	
def readLabelsCAGI4(labelFile):
	ifp = open(labelFile)
	lines = ifp.readlines()
	#print lines
	#print len(lines)
	i = 1
	db = {}
	while i < len(lines):
		tmp = lines[i].strip().split("\t")
		#print tmp
		if len(tmp[1]) == 0:
			db[tmp[0]] = 0
		else:
			db[tmp[0]] = 1
		i+=1
	print ("Found %d labels" % len(db.keys()))
	return db	#{NUM:label}

def readLabelsCAGI3(f):
	ifp = open(f)
	lines = ifp.readlines()
	#print lines
	#print len(lines)
	i = 1
	db = {}
	while i < len(lines):
		tmp = lines[i].strip().split("\t")
		#print tmp
		db[tmp[-1][3:]] = getLabel3(tmp[2])
		i+=1
	print ("Found %d labels" % len(db.keys()))
	return db	#{NUM:label}

def readLabelsCAGI2(f):
	ifp = open(f)
	lines = ifp.readlines()
	#print lines
	#print len(lines)
	i = 1
	db = {}
	while i < len(lines):
		tmp = lines[i].strip().split("\t")
		db["CAGI-"+tmp[0]] = getLabel(tmp[1])
		i+=1
	print ("Found %d labels" % len(db.keys()))
	return db	#{cagi-NUM:label}

def getLabel3(v):
	if "CD".lower() in v.lower():
		return 1
	elif "healthy" in v.lower():
		return 0
	else:
		raise Exception("ERROR: unexpected string: %s" % v)

def getLabel(v):
	if "Crohn's patient".lower() in v.lower():
		return 1
	elif "healthy" in v.lower():
		return 0
	else:
		raise Exception("ERROR: unexpected string: %s" % v)

def getScoresSVR(pred, real, threshold=None, invert = False, PRINT = False, CURVES = False, SAVEFIG=None):
	import math
	if len(pred) != len(real):
		raise Exception("ERROR: input vectors have differente len!")
	if PRINT:
		print ("Computing scores for %d predictions" % len(pred)	)	
	from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
	
	if CURVES or SAVEFIG != None:
		import matplotlib.pyplot as plt		
		precision, recall, thresholds = precision_recall_curve(real, pred)
		#plt.plot(recall, precision)
		#plt.show()
		fpr, tpr, _ = roc_curve(real, pred)		
		fig, (ax1, ax2) = plt.subplots(figsize=[10.0, 5], ncols=2)
		ax1.set_ylabel("Precision")
		ax1.set_xlabel("Recall")
		ax1.set_title("PR curve")
		ax1.set_xlim(0,1)
		ax1.set_ylim(0,1)
		ax1.plot(recall, precision)
		ax1.grid()
		ax2.plot(fpr, tpr)
		ax2.set_ylim(0,1)
		ax2.set_xlim(0,1)
		ax2.plot([0,1],[0,1],"--",c="grey",)
		ax2.set_xlabel("FPR")
		ax2.set_ylabel("TPR")
		ax2.set_title("ROC curve")
		ax2.grid()
		if SAVEFIG != None:
			plt.savefig(SAVEFIG, dpi=400)
		if CURVES == True:
			plt.show()
		#plt.clf()
		
	fpr, tpr, thresholds = roc_curve(real, pred)
	auprc = average_precision_score(real, pred)
	aucScore = auc(fpr, tpr)		
	i = 0
	r = []
	while i < len(fpr):
		r.append((fpr[i], tpr[i], thresholds[i]))
		i+=1	
	ts = sorted(r, key=lambda x:(1.0-x[0]+x[1]), reverse=True)[:3]	
	#if PRINT:
	#	print ts
	if threshold == None:
		if PRINT:
			print( " > Best threshold: " + str(ts[0][2]))
		threshold = ts[0][2]
	i = 0
	confusionMatrix = {}
	confusionMatrix["TP"] = confusionMatrix.get("TP", 0)
	confusionMatrix["FP"] = confusionMatrix.get("FP", 0)
	confusionMatrix["FN"] = confusionMatrix.get("FN", 0)
	confusionMatrix["TN"] = confusionMatrix.get("TN", 0)
	if invert == True:
		while i < len(real):
			if float(pred[i])>=threshold and (real[i]==0):
				confusionMatrix["TN"] = confusionMatrix.get("TN", 0) + 1
			if float(pred[i])>=threshold and real[i]==1:
				confusionMatrix["FN"] = confusionMatrix.get("FN", 0) + 1
			if float(pred[i])<=threshold and real[i]==1:
				confusionMatrix["TP"] = confusionMatrix.get("TP", 0) + 1
			if float(pred[i])<=threshold and real[i]==0:
				confusionMatrix["FP"] = confusionMatrix.get("FP", 0) + 1
			i += 1
	else:
		while i < len(real):
			if float(pred[i])<=threshold and (real[i]==0):
				confusionMatrix["TN"] = confusionMatrix.get("TN", 0) + 1
			if float(pred[i])<=threshold and real[i]==1:
				confusionMatrix["FN"] = confusionMatrix.get("FN", 0) + 1
			if float(pred[i])>=threshold and real[i]==1:
				confusionMatrix["TP"] = confusionMatrix.get("TP", 0) + 1
			if float(pred[i])>=threshold and real[i]==0:
				confusionMatrix["FP"] = confusionMatrix.get("FP", 0) + 1
			i += 1
	#print "--------------------------------------------"
	#print confusionMatrix["TN"],confusionMatrix["FN"],confusionMatrix["TP"],confusionMatrix["FP"]
	if PRINT:
		print ("      | DEL         | NEUT             |")
		print ("DEL   | TP: %d   | FP: %d  |" % (confusionMatrix["TP"], confusionMatrix["FP"] ))
		print ("NEUT  | FN: %d   | TN: %d  |" % (confusionMatrix["FN"], confusionMatrix["TN"])	)
	
	sen = (confusionMatrix["TP"]/max(0.00001,float((confusionMatrix["TP"] + confusionMatrix["FN"]))))
	spe = (confusionMatrix["TN"]/max(0.00001,float((confusionMatrix["TN"] + confusionMatrix["FP"]))))
	acc =  (confusionMatrix["TP"] + confusionMatrix["TN"])/max(0.00001,float((sum(confusionMatrix.values()))))
	bac = (0.5*((confusionMatrix["TP"]/max(0.00001,float((confusionMatrix["TP"] + confusionMatrix["FN"])))+(confusionMatrix["TN"]/max(0.00001,float((confusionMatrix["TN"] + confusionMatrix["FP"])))))))
	inf =((confusionMatrix["TP"]/max(0.00001,float((confusionMatrix["TP"] + confusionMatrix["FN"])))+(confusionMatrix["TN"]/max(0.00001,float((confusionMatrix["TN"] + confusionMatrix["FN"])))-1.0)))
	pre =(confusionMatrix["TP"]/max(0.00001,float((confusionMatrix["TP"] + confusionMatrix["FP"]))))
	mcc =	( ((confusionMatrix["TP"] * confusionMatrix["TN"])-(confusionMatrix["FN"] * confusionMatrix["FP"])) / max(0.00001,math.sqrt((confusionMatrix["TP"]+confusionMatrix["FP"])*(confusionMatrix["TP"]+confusionMatrix["FN"])*(confusionMatrix["TN"]+confusionMatrix["FP"])*(confusionMatrix["TN"]+confusionMatrix["FN"]))) )  
	
	if PRINT:
		print( "\nSen = %3.3f" % sen)
		print( "Spe = %3.3f" %  spe)
		print( "Acc = %3.3f " % acc)
		print( "Bac = %3.3f" %  bac)
		#print "Inf = %3.3f" % inf
		print ("Pre = %3.3f" %  pre)
		print ("MCC = %3.3f" % mcc)
		print ("#AUC = %3.3f" % aucScore)
		print ("#AUPRC= %3.3f" % auprc)
		print ("--------------------------------------------"	)
	
	return sen, spe, acc, bac, pre, mcc, aucScore, auprc

def bestLprecision(yp, y, L):
	tmp = []
	assert len(yp) == len(y)
	assert len(yp) > L
	i = 0
	while i < len(yp):
		tmp.append((yp[i], y[i]))
		i+=1
	tmp = sorted(tmp, key = lambda x:x[0], reverse=True)[:L]
	tmpy = []
	i = 0
	while i < L:
		tmpy.append(tmp[i][1])
		i+=1
	print ("Best %d precision: %3.3f" % (L, sum(tmpy)/float(L)))

def main():
	#print readLabelsCAGI2("../dataCAGI/CAGI2011_Crohns_labels/CAGI2011_CrohnsDisease_resultskey.txt")
	#print readLabelsCAGI3("../dataCAGI/CAGI2013_Crohns_labels/CAGI3_crohns_labels")
	print (readLabelsCAGI4("../../dataCAGI/labels/CAGI4labels.txt"))
	return 0

if __name__ == '__main__':
	main()
