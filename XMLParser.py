########################################################################
#
# File:   XMLParser.py
# Author: Tom Wilmots
# Date:   January 23, 2017
#
# Written for ENGR 90
#
########################################################################
#
# This script is a simple XML parser for our stroke data

import numpy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os 

cwd = os.getcwd()
ext = '/Data/lineStrokes/a01/a01-000/a01-000u-01.xml'
tree = ET.parse(cwd + ext)

root = tree.getroot()

# for child in root: # we should have two child elements in root: <WhiteboardDescription> and <StrokeSet>
# 	print child.tag
# 	for subchild in child: # We should have 4 subchild elements for element <WhiteboardDescription> and 1 for <StrokeSet>
# 		print "These are our subchild elements: ", subchild.tag
# 		for attribute in subchild:
# 			data = attribute.attrib
# print root[1][0][0].attrib

xpoints = []
ypoints = []
for point in root.iter('Point'):
	x = point.get('x')
	x = int(x)
	xpoints.append(x)
	y = point.get('y')
	y = int(y)
	ypoints.append(y)

for i in range(len(ypoints)-1):
	ypoints[i] = 1376 - ypoints[i]

plt.plot(xpoints,ypoints,'b.')
plt.show()

