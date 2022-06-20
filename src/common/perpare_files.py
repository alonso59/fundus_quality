import glob, os
# Current directory
this_dir = os.path.dirname(os.path.realpath(__file__))
current_dir = "images"
# Percentage of images to be used for the test set
percentage_test = 20
# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')  
file_test = open('val.txt', 'w')
# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
	title, ext = os.path.splitext(os.path.basename(pathAndFilename))
	if counter == index_test:
		counter = 1
		file_test.write(this_dir + current_dir + "/" + title + '.jpg' + "\n") #"/" +
	else:
		file_train.write(this_dir + current_dir + "/" + title + '.jpg' + "\n")
		counter = counter + 1
