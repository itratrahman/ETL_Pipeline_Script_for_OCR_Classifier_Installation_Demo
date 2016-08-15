
'''
Created on 2016-may

Title: Optical Character Recognition
Company: Southeast Bank Limited
@author: Itrat Rahman
Position: Software Engineering Intern
Department: IT
Objective: This source code contains just the early version of my OCR fucntions needed for character segmentation from character extraction source files. 
The original source file in my work of OCR software contains much more updated, robust, and error tolerant functions; and contains much more functions than these. 
These functions are fine for character segmentation purpose as used by classifier installation file on character extraction source files, where each characters are finely separated by 3-4 spaces. 
In real world scenario characters are loosely separated and may be conjoining in many cases; character segmentation in such case needs further processing. 
'''




##Import statements
import cv2 #importing the OpenCV Library
import numpy as np #importing numpy
import math
from sklearn import svm
from sklearn.externals import joblib
import os




##Mapping Dictionary
mapper = {
	0:65,
	1:66,
	2:67,
	3:68,
	4:69,
	5:70,
	6:71,
	7:72,
	8:73,
	9:74,
	10:75,
	11:76,
	12:77,
	13:78,
	14:79,
	15:80,
	16:81,
	17:82,
	18:83,
	19:84,
	20:85,
	21:86,
	22:87,
	23:88,
	24:89,
	25:90,
	26:97,
	27:98,
	28:99,
	29:100,
	30:101,
	31:102,
	32:103,
	33:104,
	34:105,
	35:106,
	36:107,
	37:108,
	38:109,
	39:110,
	40:111,
	41:112,
	42:113,
	43:114,
	44:115,
	45:116,
	46:117,
	47:118,
	48:119,
	49:120,
	50:121,
	51:122,
	52:48,
	53:49,
	54:50,
	55:51,
	56:52,
	57:53,
	58:54,
	59:55,
	60:56,
	61:57,
	62:46,
	63:44,
	64:59,
	65:58,
	66:63,
	67:47,
	68:92,
	69:39,
	70:33,
	71:64,
	72:35,
	73:36,
	74:37,
	75:94,
	76:38,
	77:42,
	78:40,
	79:41,
	80:45,
	81:43,
	82:61,
	83:91,
	84:93,
	85:60,
	86:62,
}




def line_removal(image):


	image_copy = image.copy()
	
	laplacian1 = cv2.Laplacian(image_copy.copy(),cv2.CV_8UC1) 

	minLineLength1 = 300
	maxLineGap1 = 4	
	lines1 = cv2.HoughLinesP(laplacian1,1,np.pi/180,100,minLineLength = minLineLength1,maxLineGap = maxLineGap1)

	laplacian2 = cv2.Laplacian(image_copy.copy(),cv2.CV_8UC1)

	minLineLength2 = int(minLineLength1/2)
	maxLineGap2 = 4
	lines2 = cv2.HoughLinesP(laplacian2,1,np.pi/180,100,minLineLength = minLineLength2,maxLineGap = maxLineGap2)


	if lines1 is not None:

		for line in lines1:

			for x1,y1,x2,y2 in line:
			
				if abs(x2-x1)>abs(y2-y1):
			
					cv2.line(image_copy,(x1,y1),(x2,y2),(255,255,255),3)

					
	if lines2 is not None:

		for line in lines2:

			for x1,y1,x2,y2 in line:
			
				if abs(x2-x1)<abs(y2-y1):
			
					cv2.line(image_copy,(x1,y1),(x2,y2),(255,255,255),3)

	return image_copy
	

	
	
def under_line_removal(image, line):

	image_copy = image.copy()
	
	inspected_ys = []
	
	threshold = 30
	
	for i in range(len(line)/2):
	
		index = 2*i
		
		line_height = line[index+1]-line[index]	

		number = int(0.35*line_height)
		
		ys = [line[index+1]-n for n in range(number)]
		
		inspected_ys = inspected_ys + ys
		
	for i in inspected_ys:
	
		lines = 0
		
		count = 0
		
		indexes = np.where(image_copy[i,:] == 0)[0]
				
		for m in range(len(indexes)-1):
		
			if (indexes[m+1]-indexes[m]) == 1:
			
				count += 1
				
			elif count>threshold:
	
				lines += 1

				count = 0
			
			else:
			
				count = 0
				
		if lines >= 1:			
			
			image_copy[i,:] = 255	

	return image_copy
				



def line_segmentation(image,histogram):

	line = []	
	
	minimum_of_histogram = min(histogram)

	non_zero_tracking_phase = False
	non_zero_counter = 0

	for i in range(image.shape[0]):

		if histogram[i] > minimum_of_histogram:
		
			if non_zero_tracking_phase == False:
			
				non_zero_counter = 1
				
				
			else:
			
				non_zero_counter += 1				

			non_zero_tracking_phase = True
		
		else:
		
			if non_zero_tracking_phase == True:
			
				if non_zero_counter > 5 and ((float(image.shape[0])/float(non_zero_counter))>11.0):
				
					line.append(i-non_zero_counter)
				
					line.append(i)
					
				non_zero_counter = 0

			non_zero_tracking_phase = False
			
	line.sort()
	
	return line
	
	
	
	
def line_heights(line):

	lineHeights = []

	for i in range(len(line)/2):
	
		index = 2*i
		
		lineHeights.append(line[index+1]-line[index])
		
	return lineHeights
	
	
	
	
def character_segmentation(image, histogram, line_height):

	character = []
	
	non_zero_tracking_phase = False
	non_zero_counter = 0
	zero_counter = 0	
	threshold = 1
	ratio_threshold = 0.65
	
	
	for i in range(image.shape[1]):
	
		max_hist = 0
			
		if histogram[i]>0 and non_zero_tracking_phase == False:
			
			non_zero_tracking_phase = True
		
			non_zero_counter = 1
				
		elif histogram[i]>=threshold and non_zero_tracking_phase == True:
			
			non_zero_counter += 1
		
		elif histogram[i]<threshold and non_zero_tracking_phase == True:

			zero_counter += 1
			
			if zero_counter >= 2 and non_zero_counter>4:
			
				non_zero_tracking_phase = False	
					
				
				character.append(i-non_zero_counter)
				
				character.append(i-zero_counter+1)

				non_zero_counter = 0					
				zero_counter = 0
				
			else:
			
				non_zero_counter += 1				
				

	character.sort()
	
	return character
	
	
	
	
	
def character_parsing(image, line):

	characters = []
	
	no_character_index = []
	
	n = []

	for i in range(len(line)/2):

		index = 2*i

		line_image = image[line[index]:line[index+1],:]
		
		line_height = line[index+1] - line[index]
		
		vertical_hist = vertical_histogram(line_image)
		vertical_hist = np.array(vertical_hist)
		
		character = character_segmentation(line_image, vertical_hist, line_height)
		
		if len(character)>=2:
		
			characters.append(character)
			
		else:
		
			n.append(i)
			
	if len(n)>0:
			
		for i in n:
		
			no_character_index.append(2*i)
			no_character_index.append((2*i)+1)
			
		modified_line = []
			
		for i in range(len(line)):
		
			if i not in no_character_index:
			
				modified_line.append(line[i])
			
		line = modified_line		
		
	return characters, line

	
	
	
def horizontal_histogram(image):

	histogram =[]
	
	for i in range(np.shape(image)[0]):

		processed_array = np.where(image[i,:] == 0)	
		

		histogram.append(np.shape(processed_array)[1])
		
	return histogram
	
	
	
	
def vertical_histogram(image):

	histogram =[]
	
	for i in range(np.shape(image)[1]):

		processed_array = np.where(image[:,i] == 0)	

		histogram.append(np.shape(processed_array)[1])
		
	return histogram
	
	
	
	
def normal_histogram(array):

	histogram = {}
	
	for element in array:
	
		if element not in histogram:
		
			histogram[element] = 1

		else:
		
			histogram[element] += 1
			
	return histogram
	
	
	
	
def character_and_space_spacings(characters, line):

	character_spacings = []

	space_spacings = []

	for i in range(len(line)/2):

		character_spacing = []
		space_spacing = []

		for j in range(len(characters[i])/2):
			
			index1 = 2*j
			
			index2 = 2*j+1
			
			character_spacing.append(characters[i][index1+1] - characters[i][index1])
			
			if j < ((len(characters[i])/2)-1):
				space_spacing.append(characters[i][index2+1] - characters[i][index2])
				
		character_spacings.append(character_spacing)
		space_spacings.append(space_spacing)		
		
	return character_spacings, space_spacings
	
	
	
	
def line_to_space_ratio(line, space_spacings):

	lineToSpaceRatio = []
	
	threshold1 = 3.85

	for i in range(len(line)/2):
	
		index1 = 2*i
			
		vertical_spacing = line[index1+1]-line[index1]
		
		array = [round((float(vertical_spacing)/float(x)),3) if x>0 else threshold1 for x in space_spacings[i]]
				
		lineToSpaceRatio.append(array)
		
	return lineToSpaceRatio
	


	
def document_segmentation(line, space_spacings, lineHeights, lineToSpaceRatio):

	characterSegmentationIDs = []
	
	threshold1 = 3.85
	
	threshold2 = 0.5

	for i in range(len(line)/2):
		
		characterSegmentationID = []
		
		characterSegmentationID.append(0)
	
		for j in range(len(space_spacings[i])):
		
			
			if lineToSpaceRatio[i][j] < threshold1 and lineToSpaceRatio[i][j] >= threshold2:		

				characterSegmentationID.append(2)
				
			if lineToSpaceRatio[i][j] < threshold2:
			
				characterSegmentationID.append(1)
				
			if lineToSpaceRatio[i][j] >= threshold1:
			
				characterSegmentationID.append(3)
				
		characterSegmentationIDs.append(characterSegmentationID)
		
	return characterSegmentationIDs
		
	
	
	
def character_scaling(character_image):


	hh = horizontal_histogram(character_image)
	
	hh = np.array(hh)
	
	indexes = np.where(hh>0)[0]
	
	top = indexes[0]
	
	bottom = indexes[-1]
	
	character_image = character_image[top:bottom+1]
	

	a = round((32.0/np.shape(character_image)[0]),2)	
	
	b = round((32.0/np.shape(character_image)[1]),2)	
	
	if np.shape(character_image)[0]>32 or np.shape(character_image)[1]>32:
			
	# if (a*np.shape(character_image)[0])<36 or (b*np.shape(character_image)[1])<36:
	
		if a<b:	
		
			character_image = cv2.resize(character_image,None,fx=a, fy=a, interpolation = cv2.INTER_LINEAR)		
			
		else:
		
			character_image = cv2.resize(character_image,None,fx=b, fy=b, interpolation = cv2.INTER_LINEAR)	
	
	character_spacing = np.shape(character_image)[1]	
	
	vertical_spacing = np.shape(character_image)[0]
	
	horizontal_deficit = 36-character_spacing

	vertical_deficit = 36-vertical_spacing
	
	if horizontal_deficit > 0:
	
		if horizontal_deficit > 1:
		
			if horizontal_deficit%2 == 0:
			
				left_padding = right_padding = horizontal_deficit/2
				
			else:
			
				left_padding = int(horizontal_deficit/2)
				right_padding = int(horizontal_deficit/2)+1
				
		else:
		
			left_padding = 1
			right_padding = 0
	else: 
				
		left_padding = right_padding = 0
				
	horizontal_array_left = np.ones([vertical_spacing, left_padding])*255	

	if right_padding > 0:
	
		horizontal_array_right = np.ones([vertical_spacing, right_padding])*255	
		
		character_image = np.concatenate((horizontal_array_left,character_image, horizontal_array_right), axis=1)

	else:
	
		character_image = np.concatenate((horizontal_array_left,character_image), axis=1)
	
	
	
	if vertical_deficit > 0:
	
		if vertical_deficit > 1:
		
			if vertical_deficit%2 == 0:
			
				top_padding = bottom_padding = vertical_deficit/2
				
			else:
			
				top_padding = int(vertical_deficit/2)
				bottom_padding = int(vertical_deficit/2)+1
				
		else:
		
			top_padding = 1
			bottom_padding = 0
			
	else:
	
		top_padding = bottom_padding = 0
				
	vertical_array_top = np.ones([top_padding, np.shape(character_image)[1]])*255		
	
	if bottom_padding > 0:
	
		vertical_array_bottom = np.ones([bottom_padding, np.shape(character_image)[1]])*255	
		
		character_image = np.concatenate((vertical_array_top, character_image, vertical_array_bottom), axis=0)

	else:
		
		character_image = np.concatenate((vertical_array_top, character_image), axis=0)
	
	return character_image
	
		
		
		