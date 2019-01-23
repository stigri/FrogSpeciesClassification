import pyexiv2
import sys
import os
import magic
import re
from collections import defaultdict
import pickle

## Script to get inventory of what images showing which genus and species there are
## TODO: Use as guideline to write dictspnrsvl, dictspnrweight, dictspnrgps, etc.
## TODO: (Do not read metadata from exif then but from xmp data)



def inventory_exif(directory):
    dictgenusspecies = defaultdict(list)
    dictspeciesspnr = defaultdict(list)
    dictspnrpath = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith((".JPG", ".jpg")):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                if not os.path.exists(absolutimpath):
                    # print('path %s is a broken symlink' % name)
                    pass
                elif not magic.from_file(absolutimpath).startswith("JPEG image data"):
                    pass
                else:
                    ## read metadata from exif tags
                    metadata = pyexiv2.ImageMetadata(absolutimpath)
                    metadata.read()
                    tag = metadata['Exif.Photo.UserComment']
                    genus_species = tag.value
                    match = re.search(r'(\w+)\s(\w+)', genus_species)
                    genus = match.group(1)
                    ## make sure that genus sp ist saved for each genus by calling genus sp = genus genus
                    if match.group(2) == 'sp':
                        species = genus
                    else:
                        species = match.group(2)
                    # if species == 'barbonica':
                    #     print(absolutimpath)
                    tag2 = metadata['Exif.Image.ImageDescription']
                    spnr = tag2.value

                    ## create dictionaries with information:
                    ## dictgenus = key: genus, value: species
                    ## dictspecies = key: species, value: spnr
                    ## dictspnr = key: spnr, value: absolutimpath
                    if genus not in dictgenusspecies.keys():
                        ## if key genus not in dictgenus add information to all dicts
                        dictgenusspecies[genus].append(species)
                        dictspeciesspnr[species].append(spnr)
                        dictspnrpath[spnr].append(absolutimpath)
                    elif species not in dictspeciesspnr.keys():
                        ## if key species not in dictspecies add information to all dicts
                        dictgenusspecies[genus].append(species)
                        dictspeciesspnr[species].append(spnr)
                        dictspnrpath[spnr].append(absolutimpath)
                    elif spnr not in dictspnrpath.keys():
                        ## if spnr not in dictspnr add spnr to dictspecies and spnr and path to dictspnr
                        dictspeciesspnr[species].append(spnr)
                        dictspnrpath[spnr].append(absolutimpath)
                    else:
                        ## if spnr already in dictspnr, add path to dictspnr
                        dictspnrpath[spnr].append(absolutimpath)

    with open('dictinventory.pkl', 'wb') as di:
        pickle.dump([dictgenusspecies, dictspeciesspnr, dictspnrpath], di)

    ## show what genus and species, how many individuals and number of pictures per species I got
    nrpictavspec = 0
    nrpictavgen = 0
    for genus in dictgenusspecies:
        nrindgen = 0
        for species in dictgenusspecies[genus]:
            nrindgen += len(dictspeciesspnr[species])
            nrindspec = len(dictspeciesspnr[species])
            nrpict = 0
            for spnr in dictspeciesspnr[species]:
                nrpict += len(dictspnrpath[spnr])
            if nrindspec > 3:
                print('{}, {}, {}, {}'.format(genus, species, nrindspec, nrpict))
                if species != 'sp':
                    nrpictavspec += nrpict
            nrpictavgen += nrpict
    print(nrpictavspec / len(dictspeciesspnr))
    print(nrpictavgen / len(dictgenusspecies))
    print(len(dictspeciesspnr))
    print(len(dictgenusspecies))

















if len(sys.argv) != 2:
    sys.stderr.write(
        'Usage: inventory_exif.py <path to directory containing .jpg files>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]

inventory_exif(directory)