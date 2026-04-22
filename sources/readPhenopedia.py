#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  readPhenopedia.py
#  
#  Copyright 2017 Daniele Raimondi <daniele.raimondi@vub.be>
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

def readPhenopedia(f, minRef=0):
	ifp = open(f)
	line = ifp.readline()
	i = 0
	w = {}
	l = []
	while len(line) > 0:
		if "#" == line[0]:
			line = ifp.readline()
			continue
		tmp = line.split()
		if int(tmp[1]) > minRef:
			w[tmp[0]] = int(tmp[1])
			l.append(tmp[0])
		i+=1	
		line = ifp.readline()	
	print ("Found %d associated genes." % i)
	return w, l
	
def main():
	readPhenopedia("../databases/phenopediaCrohnGenes/CrohnGenes")

if __name__ == '__main__':
	main()
