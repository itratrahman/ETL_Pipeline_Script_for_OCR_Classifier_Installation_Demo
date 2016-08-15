####This source code extracts charcter imagse from a document and scales them, and geneartes equal size images of characters 

##Import statements
import cv2 #importing the OpenCV Library
import numpy as np #importing numpy
import math
import matplotlib.pyplot as plt
import pandas as pd
from Segmentation import *	


##Intooductory print statement
print "This source code extracts charcter imagse from a document and scales them, and geneartes equal size images of characters ","\n"

##Name of the directory for storing the character images
fname =  str(os.getcwd())+"\Character_Document"

##Create the directory if it doesnt exist
if not os.path.exists(fname):	
	
	os.makedirs(fname)
	
else:

	print "Directory 'Character_Document' exists in the current directory.", "\n"

no_of_images = int(raw_input('Enter the number for files stored in the "Character_Document" folder: '))

print "\n"

set_size = 87

filenames = []

counter = 0

for i in range(no_of_images):

	string = str(os.getcwd())+"\Character_Document\CharacterDocument_ ("+str(i+1)+")"+".jpg"
	
	filenames.append(string)	
			
			

##Counter variable
counter = 1
	
for n in range(no_of_images):

	filename = filenames[n]

	##Reading the image from the filename
	colorImage = cv2.imread(filename)

	##Converting from RGB to Grayscale
	grayscaleimage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

	##Applying the global thresholding to the image
	ret,image = cv2.threshold(grayscaleimage,185,255,cv2.THRESH_BINARY)
	# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)

	##Displaying the image
	# cv2.imshow('Image after Morphological Operation',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	

	##Computing horizontal histogram
	horizontal_histogram_y = horizontal_histogram(image)
	##Converrting the list to numpy array
	horizontal_histogram_y = np.array(horizontal_histogram_y)

	##Computing the line_segmentation of the image 
	#The function returns a list of lines that segment the sentences
	line = line_segmentation(image,horizontal_histogram_y)

	##Computing Character Parsing
	characters, line = character_parsing(image, line)
	
	##Exracting the heights of the lines of the document
	lineHeights = line_heights(line)
	
	##Extracting character and space spacings
	character_spacings, space_spacings = character_and_space_spacings(characters, line)

	##Extracting line to space ratios of every spaces in the document line by line
	lineToSpaceRatio = line_to_space_ratio(line, space_spacings)

	##Extracting the segmentatation id for each character int the document
	characterSegmentationIDs = document_segmentation(line, space_spacings, lineHeights, lineToSpaceRatio)
	
	image_copy = colorImage.copy()	

	# print "lenght of character_spacings: ", len(character_spacings)
	# print "lenght of characterSegmentationIDs: ", len(characterSegmentationIDs)
	
	k = None
	
	for i in range(len(line)/2):
	
	
		for j in range(len(characterSegmentationIDs[i])):
		
			index1 = 2*i
			
			index2 =  2*j
		
			if characterSegmentationIDs[i][j] == 0:			
			
				image_copy[line[index1]:line[index1+1],characters[i][index2],[0,2]] = 0
				image_copy[line[index1]:line[index1+1],characters[i][index2+1],[0,2]] = 0
				image_copy[line[index1],characters[i][index2] : characters[i][index2+1],[0,2]] = 0
				image_copy[line[index1+1],characters[i][index2] : characters[i][index2+1],[0,2]] = 0
			
			
			if characterSegmentationIDs[i][j] == 1:
			
				image_copy[line[index1]:line[index1+1],characters[i][index2],2] = 0
				image_copy[line[index1]:line[index1+1],characters[i][index2+1],2] = 0
				image_copy[line[index1],characters[i][index2] : characters[i][index2+1],2] = 0
				image_copy[line[index1+1],characters[i][index2] : characters[i][index2+1],2] = 0
			
			if characterSegmentationIDs[i][j] == 2:
			
				image_copy[line[index1]:line[index1+1],characters[i][index2],:2] = 0
				image_copy[line[index1]:line[index1+1],characters[i][index2+1],:2] = 0
				image_copy[line[index1],characters[i][index2] : characters[i][index2+1],:2] = 0
				image_copy[line[index1+1],characters[i][index2] : characters[i][index2+1],:2] = 0			
			
			if characterSegmentationIDs[i][j] == 3:
			
				image_copy[line[index1]:line[index1+1],characters[i][index2],[1,2]] = 0
				image_copy[line[index1]:line[index1+1],characters[i][index2+1],[1,2]] = 0
				image_copy[line[index1],characters[i][index2] : characters[i][index2+1],[1,2]] = 0
				image_copy[line[index1+1],characters[i][index2] : characters[i][index2+1],[1,2]] = 0
				

			if counter%set_size == 0:
			
				print "Document: ", n+1
				print "Set: ", counter/set_size
				
				
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(image_copy,str(set_size - counter%set_size),((characters[i][index2]+characters[i][index2])/2,line[index1+1]+15), font, 0.5, (0,0,255),2)	
				# cv2.imshow('Image',image_copy)
				# k = cv2.waitKey(0)
				# cv2.destroyAllWindows()
				
				if k == ord("q"):
				
					break
					
			##Incrementing the counter variable
			counter += 1					
		
				
		# cv2.imshow('Image',image_copy)
		# k = cv2.waitKey(0)
		# cv2.destroyAllWindows()
	
	
		if k == ord("q"):
			
			break
			
	# cv2.imshow('Image',image_copy)
	# k = cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite(str(os.getcwd())+"\Character_Document\Segmented_Document_"+str(n+1)+".jpg", image_copy)
	
	if k == ord("q"):
	
		break