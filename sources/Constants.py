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
def getOrderedNames(ht):
	return sorted(ht.keys())

def getOrderedNamesForPrinting(ht):
	l = getOrderedNames(ht)
	r = []
	for i in l:
		r.append(i.replace("_","\n"))
	return r

def getOrderedValues(ht):
	r = []
	nameList = getOrderedNames(ht)
	for n in nameList:
		r.append(ht[n])
	return r

TYPES = {"exonic":0, "UTR3":0, "UTR5":0, "ncRNA_exonic":0, "ncRNA_intronic":0, "upstream":0, "downstream":0, "intronic":0, "splicing":0}

TYPES_ALL = {"exonic":0, "UTR3":0, "UTR5":0, "ncRNA_exonic":0, "ncRNA_intronic":0, "upstream":0, "downstream":0, "intergenic":0, "intronic":0, "splicing":0}


