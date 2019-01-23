# Bashscript to merge all .csv files to one containing all speciesnumbers and corresponding data

# IFS (Internal Field Seperator) is used by the shell to determine how to do word splitting,
# i. e. how to recognize word boundaries.
# Inside dollared single quotes, some characters are evaluated specially i.e. \n is translated to newline.
# The following line assigns newline to the variable IFS thus new line is used for word splitting instead of 
# space, tab  and newline (IFS default). This is necessary because of whitespaces used in filenames.
IFS=$'\n'

# Find searches recursively all folders and subfolders of folder frogpictures_umi/csv for all files with 
# .csv file extension. Iterate over these files and
for file in $(find /home/stine/frogpictures_umi/csv/ -name *.csv)
do
	# read 2 and 3 collumn of file, print all collumns to mergedcsvfile.csv	
	csvtool col 2,3 $file| grep -o "UA\.201[4-5]\{1\}\.[0-9]\{4\}.*" >>/home/stine/frogpictures_umi/csv/mergedcsvfile.csv
done
