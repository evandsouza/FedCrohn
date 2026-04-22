#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  runAnnovar.py
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
import sys, os, pickle, marshal
#1		2		3	4	5	6				7				8					9					10					11			12
#Chr	Start	End	Ref	Alt	Func.refGene	Gene.refGene	GeneDetail.refGene	ExonicFunc.refGene	AAChange.refGene	pLi.refGene	pRec.refGene	
#13				14						15								16							17									18	
#pNull.refGene	Gene_full_name.refGene	Function_description.refGene	Disease_description.refGene	Tissue_specificity(Uniprot).refGene	Expression(egenetics).refGene	
#19								20				21				22				23						24			25					26			27
#Expression(GNF/Atlas).refGene	P(HI).refGene	P(rec).refGene	RVIS.refGene	RVIS_percentile.refGene	GDI.refGene	GDI-Phred.refGene	SIFT_score	SIFT_converted_rankscore	
#28			29						30							31					32						33							34					35
#SIFT_pred	Polyphen2_HDIV_score	Polyphen2_HDIV_rankscore	Polyphen2_HDIV_pred	Polyphen2_HVAR_score	Polyphen2_HVAR_rankscore	Polyphen2_HVAR_pred	LRT_score	
#36							37			38						39									40					41						42
#LRT_converted_rankscore	LRT_pred	MutationTaster_score	MutationTaster_converted_rankscore	MutationTaster_pred	MutationAssessor_score	MutationAssessor_score_rankscore	
#43						44				45							46			47				48							49				50			51 				52
#MutationAssessor_pred	FATHMM_score	FATHMM_converted_rankscore	FATHMM_pred	PROVEAN_score	PROVEAN_converted_rankscore	PROVEAN_pred	VEST3_score	VEST3_rankscore	MetaSVM_score
#53					54				55				56					57			58			59				60			61			62					63			64
#MetaSVM_rankscore	MetaSVM_pred	MetaLR_score	MetaLR_rankscore	MetaLR_pred	M-CAP_score	M-CAP_rankscore	M-CAP_pred	CADD_raw	CADD_raw_rankscore	CADD_phred	DANN_score	
#65				66						67							68						69							70			71				72
#DANN_rankscore	fathmm-MKL_coding_score	fathmm-MKL_coding_rankscore	fathmm-MKL_coding_pred	Eigen_coding_or_noncoding	Eigen-raw	Eigen-PC-raw	GenoCanyon_score	
#73							74							75									76							77			78					79
#GenoCanyon_score_rankscore	integrated_fitCons_score	integrated_fitCons_score_rankscore	integrated_confidence_value	GERP++_RS	GERP++_RS_rankscore	phyloP100way_vertebrate	
#80									81						82								83							84										85
#phyloP100way_vertebrate_rankscore	phyloP20way_mammalian	phyloP20way_mammalian_rankscore	phastCons100way_vertebrate	phastCons100way_vertebrate_rankscore	phastCons20way_mammalian
#86									87					88								89				90				91				92
#phastCons20way_mammalian_rankscore	SiPhy_29way_logOdds	SiPhy_29way_logOdds_rankscore	Interpro_domain	GTEx_V6_gene	GTEx_V6_tissue	Otherinfo


def parseAnnovarMultianno(f, missenseOnly = False, onlyRegions=["exonic"], onlySNVs = False, onlyGenes = None):
	ifp = open(f)
	ifp.readline()
	line = ifp.readline()
	db = {}
	if onlyGenes != None:
		onlyGenes = set(onlyGenes)
	while len(line) > 0:
		tmp = line.split("\t")
		crom = tmp[0]
		pos = int(tmp[1])#(int(tmp[1]), int(tmp[2]))
		#mut = tmp[9]#(tmp[3], tmp[4])
		region = tmp[5]
		
		if onlyRegions != None and not region in onlyRegions:
			line = ifp.readline()
			continue	
				
		gene = tmp[6]
		if onlyGenes != None and not gene in onlyGenes:
			line = ifp.readline()
			continue	
		
		#if gene == "HLA-A":
		#	print gene	
		vartype = tmp[8]
		if missenseOnly and not "nonsynonymous" in vartype:
			line = ifp.readline()
			continue	
		if onlySNVs and not "SNV" in vartype:
			line = ifp.readline()
			continue	
		#print region, gene			
		rec = safeCastFloat(tmp[11])
		disease = tmp[15]
		tissues = parseTiss(tmp[16])
		expr1 = parseExpr(tmp[17])
		expr2 = parseExpr(tmp[18])
		hi = safeCastFloat(tmp[19])
		rvis = safeCastFloat(tmp[21])
		gdi = safeCastFloat(tmp[23])
		metasvm = safeCastFloat(tmp[52])
		mcap = safeCastFloat(tmp[57])
		domain = tmp[88]
		gtex = parseGtex(tmp[90])	
		vest = 	safeCastFloat(tmp[49])
		#print tmp[92:]
		#raw_input()
		
		#print "crom: ",crom, " pos: ", pos, " mut: ", mut, " region: ", region, " gene: ", gene, " var: ", vartype, " rec: ", rec, " dis: ", disease, " tiss: ", tissues, " expr1: ", expr1," expr2: ", expr2, " hi: ", hi, " : ", 
		#print " rvis: ", rvis, " gdi: ", gdi, " metasvm: ", metasvm, " mcap: ", mcap, " domain: ", domain, " gtex: ", gtex
		#raw_input()
		if not crom in db:
			db[crom] = []
		#db[crom].append((pos, vest, region, gene, vartype, rec, disease, tissues, expr1, expr2, hi, rvis, gdi, metasvm, mcap, domain, gtex))
		db[crom].append((pos, vest, region, gene, vartype, rec, hi, rvis, gdi, metasvm, mcap, ))
		#print (pos, vest, region, gene, vartype, rec, hi, rvis, gdi, metasvm, mcap, )
		#if region != "exonic":
		#	print region
		line = ifp.readline()
	return db	

def parseTiss(v):
	if v == ".":
		return []
	return v.split(":")[1]

def parseExpr(v):
	if v == ".":
		return []
	return v.split(";")

def parseGtex(v):
	if v == ".":
		return []
	return v.split("|")

def safeCastStr(v):
	if v == ".":
		return 0
	return v
		
def safeCastFloat(v):
	a = None
	try:
		a = float(v)
	except:
		return 0
	return a
		
	

def main():
	ex1 = parseAnnovarMultianno("../annovarCAGI2data/CAGI-1.vcf.hg18_multianno.txt")
	return 0
	
if __name__ == '__main__':
	main()
