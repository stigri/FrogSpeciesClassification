import sys
from shutil import copyfile
import os

## Code to see how many pictures there are of each individuum. Gives an overview, does not work properly since not all
## images of the same individuum are stored in consecutive pictures. Found additional errors in ocr.

def inventory_basic(directory, csvinventory):
    prev_spnr_ocr = None
    cnt = 0
    csvfile = 'inventory.csv'

    for root, dirs, files in os.walk(directory):
        for name in sorted(files):
            if name.endswith(".spnr"):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                ##print(absolutimpath)

                ## get backup copy of each .spnr file
                copyfile(absolutimpath, absolutimpath+'.bak')
                ## open .spnr files and copy spnr and path to corresponding lists
                with open(absolutimpath) as spnr:
                    spnr_ocr = spnr.readline()
                    spnr_ocr = spnr_ocr.rstrip()
                    if prev_spnr_ocr == None:
                        cnt = 1
                        prev_spnr_ocr = spnr_ocr
                    elif spnr_ocr == prev_spnr_ocr:
                        cnt = cnt + 1
                        prev_spnr_ocr = spnr_ocr
                    else:
                        with open(csvinventory) as csv:
                            for row in csv:
                                spnr_csv = row.split(',')[0]
                                row = row.rstrip()
                                if prev_spnr_ocr == spnr_csv:
                                    if os.path.exists(csvfile):
                                        append_write = 'a'  # append if already exists
                                    else:
                                        append_write = 'w'  # make a new file if not
                                    entry = open(csvfile, append_write)
                                    entry.write(row + str(cnt) + '\n')
                                    entry.close()
                                    if cnt == 1:
                                        print(absolutimpath)
                                        print(spnr_ocr)
                        prev_spnr_ocr = spnr_ocr
                        cnt = 1






## run code by entering the directory where .spnr files are and where merged csv file is
if len(sys.argv) != 3:
    sys.stderr.write(
        'Usage: inventory_basic.py <path to directory containing .spnr files, path to inventory csv>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]
    csvinventory = sys.argv[2]


inventory_basic(directory, csvinventory)