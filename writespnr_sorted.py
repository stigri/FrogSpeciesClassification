## Code used to write spnr files for sorted pictures.
## Code looks for spnr in path to write in .spnr file. There is one folder where two different spnr are written in the
## path. Code finds first spnr, which is the wrong one for the pictures with different spnr in filename.

import sys
import os
import re



def writespnr_sorted(path):
    ## list of all spnr
    spnrlist = list()
    ## list of all path to spnr files
    pathlist = list()
    ## walk directories to find all .jpg files
    for root, dirs, files in os.walk(path):
        for name in sorted(files):
            if name.endswith((".JPG", ".jpg")):
                ## get absolutepath because of git annex
                spnr_match = re.search(r'U[A,a](201[3,4,5]{1})(\d{4})', root)
                print(root)
                ## read year and number from spnr in path
                spnr_year = spnr_match.group(1)
                spnr_number = spnr_match.group(2)
                spnr = 'UA.{}.{}'.format(spnr_year, spnr_number)
                print(spnr)
                ## open .spnr files and copy spnr and path to corresponding lists
                file = open(root + "/" + name + '.spnr', 'w')
                file.write(spnr)





## run code by entering the directory where .spnr files are and where merged csv file is
if len(sys.argv) != 2:
    sys.stderr.write(
        'Usage: speciesnumberrecognition.py <path to directory containing .jpg files>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]



writespnr_sorted(directory)