

'''
Created on 2016-July

Title: Optical Character Recognition
Company: Southeast Bank Limited
@author: Itrat Rahman
Position: Software Engineering Intern
Department: IT
Objective: This source code contains the entire data engineering ETL pipeline that extracts the character data from the documents; 
cleans and transforms the character data; stores the data in either CSV file or sqlite database; and finally carries out cross_validation 
operation to find to plot "Accuracy vs C" plot to inspect the performance of SVM classifier using different C values. 

'''




####Import statements
import cv2 #importing the OpenCV Library
import numpy as np #importing numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from Segmentation import *
import os
import csv
import sqlite3
import json




fname =  str(os.getcwd())+"\\"+"Character_Document"

##If statement to check if the directory for character documents exists
if os.path.exists(fname):

	##Print statement
	print "Directory for character documents exists, therefore proceeding with the script", "\n"
    

else:

	##Print statement
	print "Directory for character documents does not exist, creating a file named Character_Documents in the current directory, \
	please put the character document files in the directory", "\n"
	
	##Creating the directory
	os.makedirs(fname)
	
	##Quitting the script
	quit()


##Intooductory print statement
print "This source code extracts goes through entire machine learning pipeline, extracting character data from image,\
cleans the image data, and training the image data using a machine learning classifiers,\
and finally evaluating the classiifiers","\n"


##Extracting the user input which controls the execution of the script
user_input1 = raw_input("Do you want to extract character images from the documents and store them in a chosen directory (y/n): ")
print "\n"
user_input2 = raw_input("Do you want to unravell the image data and store the data in a CSV file (y/n): ")
print "\n"
user_input3 = raw_input("Do you want to unravell the image data and store the data in a sqlite database file (y/n): ")
print "\n"
user_input4 = raw_input("Do you want to do cross validation operation on training data from CSV file using SVM classifiers (y/n): ")
print "\n"
user_input5 = raw_input("Do you want to do cross validation operation on training data from database file using SVM classifiers (y/n): ")
print "\n"
user_input6 = raw_input("Do you want to save a SVM classifier with a chosen value of C by training data from CSV file (y/n): ")
print "\n"
user_input7 = raw_input("Do you want to save a SVM classifier with a chosen value of C by training data from sqlite database file (y/n): ")


##user input which controlls the execution of the script
user_input = ""


##Setting the route input according to the user inputs	
if user_input1 == "y" or user_input1 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"
	
if user_input2 == "y" or user_input2 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"

if user_input3 == "y" or user_input3 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"
	
if user_input4 == "y" or user_input4 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"
	
if user_input5 == "y" or user_input5 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"

if user_input6 == "y" or user_input6 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"
	
if user_input7 == "y" or user_input7 == "Y":
	
	user_input += "1"
	
else:

	user_input += "0"
	
##Print a new line character
print "\n"

##Set Size
set_size = 87




'''This portion parses the character images from the character image files stored in the directory 'Character_Document' and store the character images 
which act as the training data in the directory 'Characters'. The directory 'Characters' is created if it not there.'''
if user_input[0] == "1":

	##Print statement
	print "Extracting the data from the documents and rescaling the character images and storing them in a designated directory", "\n"


	##A list to store the name of the filenames
	filenames = []

	##Name of the directory for storing the character images
	fname =  str(os.getcwd())+"\Characters"
	
	##Create the directory if it doesnt exist
	if not os.path.exists(fname):	
		
		os.makedirs(fname)
		
	no_of_images = int(raw_input('Enter the number for files stored in the "Character_Document" folder: '))

	print "\n"

	##Appending the filenames to the list
	for i in range(no_of_images):

		# string = "C:\Users\User\Dropbox\Southeast Bank Limited\Test_Codes\Generating_data_and_stats_analysis\Character_Document_6\CharacterDocument_ ("+str(i+1)+")"+".jpg"
		string = str(os.getcwd())+"\Character_Document\CharacterDocument_ ("+str(i+1)+")"+".jpg"
		
		filenames.append(string)
		
	##counter variable
	counter = 1
	
	##Iterating through each image document	
	for n in range(no_of_images):

		print "Processing document "+str(n+1)+" for character image retrieval"
		
		##Extracting the filename from the list
		filename = filenames[n]

		##Reading the image from the filename
		colorImage = cv2.imread(filename)

		##Converting from RGB to Grayscale
		grayscaleimage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

		##Applying the global thresholding to the image
		ret,image = cv2.threshold(grayscaleimage,185,255,cv2.THRESH_BINARY)		

		##Computing horizontal histogram
		horizontal_histogram_y = horizontal_histogram(image)
		##Converrting the list to numpy array
		horizontal_histogram_y = np.array(horizontal_histogram_y)

		##Computing the line_segmentation of the image 
		#The function returns a list of lines that segment the sentences
		line = line_segmentation(image,horizontal_histogram_y)
		
		##Skip extracting from the current document if there is no line found in the document
		if len(line)<1:
		
			break		

		##Computing Character Parsing
		characters, line = character_parsing(image, line)

		##Iterating through each line in the image
		for i in range(len(line)/2):

			##Iterating through each character in the line under consideration
			for j in range(len(characters[i])/2):
			
				##Making a copy of the image
				image_copy = image.copy()
			
				##Index for line
				index1 = 2*i
				
				##Index for character
				index2 = 2*j
				
				##Extracting the character image
				character_image = image[line[index1]:line[index1+1],characters[i][index2] : characters[i][index2+1]]			
				
				##Scaling the character image
				character_image = character_scaling(character_image)
				
				##Saving the scaled image in the designated folder
				# string = "C:\Users\User\Dropbox\Southeast Bank Limited\Characters\character" + str(counter) + ".jpg"			
				string = str(os.getcwd())+"\Characters\character" + str(counter) + ".jpg"			
				cv2.imwrite(string, character_image)
				
				##Incrementing the counter
				counter += 1
				
	print "Stored "+str(counter-1)+" character images in the designated directory", "\n"
				
	print "Finished executing character image extraction, character image rescaling and character image storage", "\n"
				


				
'''This portion extracts the character image data stored in 'Characters', cleans the data to make the data sparse, and then stores the data row-wise 
in the CSV file 'training_data'.Execution of this step depends on if the raw character data is created in step 1.'''				
if user_input[1] == "1":

	##Print statement
	print "Executing character image extraction from the designated directory, unravelling the image data and storing the data in a CSV file", "\n"

	##Counting the number of image files
	# path, dirs, files = os.walk("C:\Users\User\Dropbox\Southeast Bank Limited\Characters").next()
	path, dirs, files = os.walk(str(os.getcwd())+"\Characters").next()
	file_count = len(files)
	print "Number of character images in the directory: ", file_count, "\n"

	##Iterating through image in a for loop
	for i in range(file_count):

		##Reading the nth character image
		# filename = "C:\Users\User\Dropbox\Southeast Bank Limited\Characters\character"+str(i+1)+".jpg"
		filename = str(os.getcwd())+"\Characters\character"+str(i+1)+".jpg"
		character_image = cv2.imread(filename,0)
		
		##Applying the global thresholding to the image
		ret,character_image = cv2.threshold(character_image,150,255,cv2.THRESH_BINARY)
		
		##Normalizing the data
		character_image = character_image *(1.0/255.0)
		
		##Extracting the rows and columns of the image
		r,c = np.shape(character_image)		
		
		##Unravelling the character_image into a vector
		character_image = character_image.reshape(r*c)

		##Converting the numpy vector into a list vector
		character_image = character_image.tolist()
		
		##Appending the vector to a csv file	
		with open(r'training_data.csv', 'ab') as f:
			writer = csv.writer(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
			writer.writerow(character_image)
		
		##Creating the obsolete first row to be used as index for the pandas csv read		
		if i == 0:

			##Appending the vector to a csv file	
			with open(r'training_data.csv', 'ab') as f:
				writer = csv.writer(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
				writer.writerow(character_image)
		
	print "Finished reshaping the character images and generating training data","\n"


	
	
'''This portion extracts the character image data stored in 'Characters', cleans the data to make the data sparse, and then stores the data
 in the sqlite database file 'CharacterData.sqlite' in the form of JSON. Execution of this step depends on if the raw character data is created in step 1.'''
if user_input[2] == "1":

	##Print statement
	print "Executing character image extraction from the designated directory, unravelling the image data and storing the data in a sqlite database file", "\n"

	##Counting the number of image files
	# path, dirs, files = os.walk("C:\Users\User\Dropbox\Southeast Bank Limited\Characters").next()
	path, dirs, files = os.walk(str(os.getcwd())+"\Characters").next()
	file_count = len(files)
	print "Number of character images in the directory: ", file_count, "\n"
	
	##Forming a relational database by the name of "rosterdb"
	conn = sqlite3.connect('CharacterData.sqlite')

	##Forming the database cursor
	cur = conn.cursor()

	##SQL statement for creating the table
	string = '''DROP TABLE IF EXISTS Character_Data;
	CREATE TABLE Character_Data(
		
		id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
		
		data TEXT
	)
	'''	

	##SQL script for table setup
	cur.executescript(string)

	##Iterating through image in a for loop
	for i in range(file_count):

		##Reading the nth character image
		# filename = "C:\Users\User\Dropbox\Southeast Bank Limited\Characters\character"+str(i+1)+".jpg"
		filename = str(os.getcwd())+"\Characters\character"+str(i+1)+".jpg"
		character_image = cv2.imread(filename,0)
		
		##Applying the global thresholding to the image
		ret,character_image = cv2.threshold(character_image,150,255,cv2.THRESH_BINARY)
		
		##Normalizing the data
		character_image = character_image *(1.0/255.0)
		
		##Extracting the rows and columns of the image
		r,c = np.shape(character_image)		
		
		##Unravelling the character_image into a vector
		character_image = character_image.reshape(r*c)

		##Converting the numpy vector into a list vector
		character_image = character_image.tolist()
		
		##Converting the data to JSON
		character_image = json.dumps(character_image)
		
		##Inserting the character data into the database file
		cur.execute('''INSERT INTO Character_Data (data) 
            VALUES (?)''', (character_image,) )	

	##Committing the update into database
	conn.commit() 
	
	##Closing the database cursor
	cur.close()
		
	print "Finished reshaping the character images and generating training database file","\n"


	
	

'''This portion trains SVM classifiers on the data from CSV file 'training_data' using different C values. This portion takes the user input of start value, end value, and 
increment of the C values. Finally it generates an image file containing 'Accuracy vs C value' plot. Execution of this step depends on if the CSV file containing data is created in step 2.'''	
if user_input[3] == "1":
	
	##Filename of designated CSV file which stores the character data
	fname = str(os.getcwd())+"\\"+"training_data.csv"

	
	##If statement to check if the classifier file already exists
	if os.path.isfile(fname):

		##Print statement on the event if the classifier file is found
		print "The CSV file containing character data exists so starting cross validation operation on training data using SVM classifiers using different C values.", "\n"
		
		##Reading the CSV file
		character_data = pd.read_csv("training_data.csv", delimiter=',',header='infer', names=None, index_col=None, usecols=None)

		##Calculating the rounded number of character set that can accomodated
		no_of_sets = int((character_data.shape[0])/set_size)
		
		print "Number of character sets: ", no_of_sets, "\n"

		##A list to store the label vector of the data
		labels = []
		
		##Creating the labels for the data
		for i in range(no_of_sets):

			j = 0

			for j in range(set_size):
			
				labels.append(j)

		##Converting the array to pandas dataframe
		labels = pd.DataFrame(labels)

		##Concatenating the labels to the data frame
		character_data = pd.concat([character_data, labels], axis=1)

		##Converting the pandas dataframe into numpy array
		character_data= character_data.as_matrix(columns=None)

		##Extracting feature vector
		features = character_data[:,0:-1]

		##Extracting target variables
		target_variables = character_data[:,-1]

		##Train/Test split
		feature_train, feature_test, target_train, targer_test = train_test_split(features, target_variables, test_size = 0.20, random_state = 10)
		
		##User input for start, end and increment value of C
		start_value = int(raw_input("Enter the start value of C: "))
		end_value = int(raw_input("Enter the end value of C: "))
		increment = int(raw_input("Enter the increment value of C: "))
		print "\n"
		
		##start value, end value and increment for the range of C values
		# start_value, end_value, increment = 100, 300, 5

		##A list to store the C values for testing
		c = [i for i in range(start_value,end_value+increment,increment)]

		##A list to store the accuracies with different C values for Support Vector Machine
		accuracies = []

		##Extracting the accuracies of the SVM model with different C values
		for i in range(len(c)):

			##Print statement
			print "Testing with C value: ", c[i]

			##Instantiating a suppport vector machine classifier
			model = svm.SVC(C = c[i], kernel = "rbf")

			##Training the classifier
			model.fit(feature_train,target_train)

			##Creating the prediction vector
			predictions = model.predict(feature_test)
			
			##Calculating the accuracy score of the classifier
			accuracy = accuracy_score(targer_test, predictions)
			
			##Appending to the designated list
			accuracies.append(accuracy)

			##Evaluating  the accuracy
			print "Accuracy of the classifier with C(",str(c[i]),"): ", accuracy
			
		##Storing the information into a pandas dataframe
		cross_validation_info = pd.DataFrame({'Accuracy':accuracies,'C':c})	

		##Storing the cross validation info into CSV file
		cross_validation_info.to_csv('cross_validation_info.csv', index = False, header = True)
		
		##Print statement
		print "\n", "Completed cross validation operation on data using multiple C values"
			
		##Plotting the accuracy vs C plot and saving the plot
		plt.plot(c, accuracies)
		plt.xlabel('C')
		plt.ylabel('Accuracy')
		plt.title('Accuracy vs C plot')
		plt.grid(True)
		string = "Accuracy_Vs_C_Plot.png"
		plt.savefig(string,format = 'png')
		plt.close()		
		
	else:

		##Print statement on the event if the classifier file is not found
		print "The designated CSV file does not exists so exiting the current part of script routine", "\n"	

	
	
	
'''This portion trains SVM classifiers on the data from sqlite database file 'training_data' using different C values. This portion takes the user input of start value, 
end value, and increment of the C values. Finally it generates an image file containing 'Accuracy vs C value' plot. Beware the script might generate memory error if the data file is too large. 
Execution of this step depends on if the sqlite database file containing data is created in step 3.'''
if user_input[4] == "1":

	##Filename of designated CSV file which stores the character data
	fname = str(os.getcwd())+"\\"+"\CharacterData.sqlite"

	##If statement to check if the classifier file already exists
	if os.path.isfile(fname):

		##Print statement on the event if the classifier file is found
		print "The sqlite database file containing character data exists so starting cross validation operation on training data using SVM classifiers using different C values.", "\n"
		
		##Forming a relational database by the name of "rosterdb"
		conn = sqlite3.connect('CharacterData.sqlite')

		##Forming the database cursor
		cur = conn.cursor()
		
		##A list to store the character data
		character_data = []
		
		##SQL command to extract all the data fromt the locations table
		cur.execute('SELECT * FROM Character_Data')	
		
		##Iterating through each row returned by the SQL Query
		for row in cur:
		
			##Extracting the data
			data = str(row[1])
			
			##Converting the JSON to list
			data = json.loads(data)
			
			##Appending the character data to the designated list
			character_data.append(data)	

		##Converting the list to numpy array
		features = np.array(character_data)

		##Calculating the rounded number of character set that can accomodated
		no_of_sets = int((features.shape[0])/set_size)
		
		# print "No of sets: ", no_of_sets, "\n"		

		##A list to store the label vector of the data
		labels = []
		
		##Creating the labels for the data
		for i in range(no_of_sets):

			j = 0

			for j in range(set_size):
			
				labels.append(j)
				
		target_variables = np.array(labels)
		
		# print "Shape of character_Data: ", np.shape(features)
		
		# print "Shape of target_variables: ", np.shape(target_variables)
		

		##Splitting the data for testing purpose
		feature_train, feature_test, target_train, targer_test = train_test_split(features, target_variables, test_size = 0.20, random_state = 10)
		
		##User input for start, end and increment value of C
		start_value = int(raw_input("Enter the start value of C: "))
		end_value = int(raw_input("Enter the end value of C: "))
		increment = int(raw_input("Enter the increment value of C: "))
		print "\n"
		
		##start value, end value and increment for the range of C values
		# start_value, end_value, increment = 100, 400, 5

		##A list to store the C values for testing
		c = [i for i in range(start_value,end_value+increment,increment)]

		##A list to store the accuracies with different C values for Support Vector Machine
		accuracies = []
		
		##A list to store the accuracies with different C values for Support Vector Machine
		accuracies = []

		##Extracting the accuracies of the SVM model with different C values
		for i in range(len(c)):

			##Print statement
			print "Testing with C value: ", c[i]

			##Instantiating a suppport vector machine classifier
			model = svm.SVC(C = c[i], kernel = "rbf")

			##Training the classifier
			model.fit(feature_train,target_train)

			##Creating the prediction vector
			predictions = model.predict(feature_test)
			
			##Calculating the accuracy score of the classifier
			accuracy = accuracy_score(targer_test, predictions)
			
			##Appending to the designated list
			accuracies.append(accuracy)

			##Evaluating  the accuracy
			print "Accuracy of the classifier with C(",str(c[i]),"): ", accuracy
			
		##Storing the information into a pandas dataframe
		cross_validation_info = pd.DataFrame({'Accuracy':accuracies,'C':c})	

		##Storing the cross validation info into CSV file
		cross_validation_info.to_csv('cross_validation_info.csv', index = False, header = True)
		
		##Print statement
		print "\n", "Completed cross validation operation on data using multiple C values"
			
		##Plotting the accuracy vs C plot and saving the plot
		plt.plot(c, accuracies)
		plt.xlabel('C')
		plt.ylabel('Accuracy')
		plt.title('Accuracy vs C plot')
		plt.grid(True)
		string = "Accuracy_Vs_C_Plot.png"
		plt.savefig(string,format = 'png')
		plt.close()
		
	else:

		##Print statement on the event if the classifier file is not found
		print "The designated sqlite database file does not exists so exiting the current part of script routine", "\n"	

	


'''This portion generates SVM classifier files in the directory 'classifier' with a chosen value of C from the user, by extracting data from the CSV file 'training_data'. 
The directory 'classifier' is created if it not there. Execution of this step depends on if the CSV file containing data is created in step 2.'''
if user_input[5] == "1":


	##Filename of the file the contains the trained classifier object
	fname1 = str(os.getcwd())+'\classifier\SVM_Classifier.pkl'

	##Printing t
	print "\n"

	##If statement to check if the classifier file already exists
	if os.path.isfile(fname1):

		##Print statement on the event if the classifier file is found
		print "The classifier file exists so replacing the existing classifier.", "\n"
		
		
	else:

		##Print statement on the event if the classifier file is not foung
		print "The classifier file does not exists so trianing the classifier first", "\n"
		
		dir =  str(os.getcwd())+"\classifier"

		##If statement to check if the directory for classifer exists
		if os.path.exists(dir):

			##Print statement
			print "Directory for classifer exists, therefore proceeding with the script", "\n"
			

		else:

			##Print statement
			print "Directory for classifer does not exist, creating a file named classifer in the current direcotry", "\n"
			
			##Creating the directory
			os.makedirs(dir)
			
	##Filename of designated CSV file which stores the character data
	fname2 = str(os.getcwd())+"\\"+"training_data.csv"

	
	##If statement to check if the classifier file already exists
	if os.path.isfile(fname2):
	
		##Print statement on the event if the classifier file is found
		print "The CSV file containing character data exists so saving a SVM classifier with a chosen value of C.", "\n"

		##Reading the CSV file
		character_data = pd.read_csv("training_data.csv", delimiter=',',header='infer', names=None, index_col=None, usecols=None)

		##Calculating the rounded number of character set that can accomodated
		no_of_sets = int((character_data.shape[0])/set_size)

		##A list to store the label vector of the data
		labels = []

		##Creating the labels for the data
		for i in range(no_of_sets):

			j = 0

			for j in range(set_size):

				labels.append(int(j))

		##Converting the array to pandas dataframe
		labels = pd.DataFrame(labels)

		##Concatenating the labels to the data frame
		character_data = pd.concat([character_data, labels], axis=1)

		##Converting the pandas dataframe into numpy array
		character_data= character_data.as_matrix(columns=None)

		##Extracting feature vector
		features = character_data[:,0:-1]
		# features = np.array(features)

		# print "Shape of features: ", np.shape(features)

		##Extracting target variables
		target_variables = character_data[:,-1]
		# target_variables = np.array(target_variables)

		# print "Shape of target_variables: ", np.shape(target_variables)

		##Splitting the data for testing purpose
		feature_train, feature_test, target_train, targer_test = train_test_split(features, target_variables, test_size = 0.00, random_state = 10)

		##User input to store the value of C
		input = raw_input("Enter the value of C you want to use for Support Vector Machine: ")
		
		print "\n"
		
		##If the user inputs a digit then store the digit or take the default value of c
		if input.isdigit():
		
			c = int(input)
		
		else:
		
			c = 105
			
		##Print statement at the start of classification
		print "Starting SVM Classficaton using a C value of ", c

		#Instantiating a suppport vector machine classifier
		model = svm.SVC(C = c, kernel = "rbf")

		##Training the classifier
		model.fit(feature_train,target_train)


		##Saving the trained classifier object
		joblib.dump(model, fname1) 
		
	else:
	
		##Print statement on the event if the classifier file is not found
		print "The designated CSV file does not exists so exiting the current part of script routine", "\n"	
	

	
		
'''This portion generates SVM classifier files in the directory 'classifier' with a chosen value of C from the user, by extracting data from the sqlite database file 
'CharacterData.sqlite'. The directory 'classifier' is created if it not there. Beware the script might generate memory error if the data file is too large. 
Execution of this step depends on if the sqlite database file containing data is created in step 3.'''
if user_input[6] == "1":


	##Filename of the file the contains the trained classifier object
	fname1 = str(os.getcwd())+'\classifier\SVM_Classifier.pkl'

	##Printing t
	print "\n"

	##If statement to check if the classifier file already exists
	if os.path.isfile(fname1):

		##Print statement on the event if the classifier file is found
		print "The classifier file exists so replacing the existing classifier.", "\n"
		
		
	else:

		##Print statement on the event if the classifier file is not foung
		print "The classifier file does not exists so trianing the classifier first", "\n"
		
		dir =  str(os.getcwd())+"\classifier"

		##If statement to check if the directory for classifer exists
		if os.path.exists(dir):

			##Print statement
			print "Directory for classifer exists, therefore proceeding with the script", "\n"
			

		else:

			##Print statement
			print "Directory for classifer does not exist, creating a file named classifer in the current direcotry", "\n"
			
			##Creating the directory
			os.makedirs(dir)
			
	##Filename of designated CSV file which stores the character data
	fname2 = str(os.getcwd())+"\\"+"\CharacterData.sqlite"

	
	##If statement to check if the classifier file already exists
	if os.path.isfile(fname2):

		##Print statement on the event if the classifier file is found
		print "The sqlite database file containing character data exists so saving a SVM classifier with a chosen value of C.", "\n"
		
		##Forming a relational database by the name of "rosterdb"
		conn = sqlite3.connect('CharacterData.sqlite')

		##Forming the database cursor
		cur = conn.cursor()
		
		##A list to store the character data
		character_data = []
		
		##SQL command to extract all the data fromt the locations table
		cur.execute('SELECT * FROM Character_Data')	
		
		##Iterating through each row returned by the SQL Query
		for row in cur:
		
			##Extracting the data
			data = str(row[1])
			
			##Converting the JSON to list
			data = json.loads(data)
			
			##Appending the character data to the designated list
			character_data.append(data)	

		##Converting the list to numpy array
		features = np.array(character_data)

		##Calculating the rounded number of character set that can accomodated
		no_of_sets = int((features.shape[0])/set_size)
		
		# print "No of sets: ", no_of_sets, "\n"q		

		##A list to store the label vector of the data
		labels = []
		
		##Creating the labels for the data
		for i in range(no_of_sets):

			j = 0

			for j in range(set_size):
			
				labels.append(j)
				
		target_variables = np.array(labels)
		
		# print "Shape of character_Data: ", np.shape(features)
		
		# print "Shape of target_variables: ", np.shape(target_variables)
		

		##Splitting the data for testing purpose
		feature_train, feature_test, target_train, targer_test = train_test_split(features, target_variables, test_size = 0.00, random_state = 10)

		##User input to store the value of C
		input = raw_input("Enter the value of C you want to use for Support Vector Machine: ")
		
		print "\n"
		
		##If the user inputs a digit then store the digit or take the default value of c
		if input.isdigit():
		
			c = int(input)
		
		else:
		
			c = 105
			
		##Print statement at the start of classification
		print "Starting SVM Classficaton using a C value of ", c

		#Instantiating a suppport vector machine classifier
		model = svm.SVC(C = c, kernel = "rbf")

		##Training the classifier
		model.fit(feature_train,target_train)


		##Saving the trained classifier object
		joblib.dump(model, fname1) 
		
	else:
	
		##Print statement on the event if the classifier file is not found
		print "The designated sqlite database file does not exists so exiting the current part of script routine", "\n"	
	