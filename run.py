from doct_tools import Segmenter
import glob,os,sys

flist = glob.glob('/home/rjonnal/data/Dropbox/Share/hroct_data/2017.07.06_registered_projections/AVG*.tif')
flist.sort()

for f in flist:
    partial = os.path.split(f)[1] # split the filename into a list [directory,filename] and get the second
    tokens = partial.split('_') # tokenize the filename using underscores
    pupil_position = int(tokens[-2])

    s = Segmenter(f)
    
