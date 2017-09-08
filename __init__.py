from matplotlib import pyplot as plt
from clicky import collector
import numpy as np
import sys,os
from scipy.misc import imread
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

class Segmenter:

    def __init__(self,fn):

        tag = os.path.splitext(fn)[0]
        
        im = imread(fn)
        prof = im.mean(axis=1)
        zmin = np.argmax(prof)-100
        zmax = np.argmax(prof)+100
        subim = im[zmin:zmax,:]
        kernel = np.ones((5,5))
        subim = fftconvolve(subim,kernel,mode='same')
        subim = np.log(subim)
        ssy,ssx = subim.shape
        y1,y2 = ssy//2-50,ssy//2+50
        x1,x2 = ssx//2-50,ssx//2+50
        cmin,cmax = np.percentile(subim[y1:y2,x1:x2],(0,99.5))
        subim = (subim-cmin)/(cmax-cmin)
        subim = (subim.clip(0,1)*255.0).astype(np.uint8)
        xclicks,yclicks,junk = collector([subim],titles=['click multiple points along IS/OS band'])

        xtemp = []
        ytemp = []

        # search a bit to make sure we've got the brightest pixel:
        rad = 1
        for xc,yc in zip(xclicks,yclicks):
            xc = int(xc)
            yc = int(yc)
            col = subim[yc-rad:yc+rad+1,xc]
            shift = np.argmax(col)-1
            yc = yc + shift
            xtemp.append(xc)
            ytemp.append(yc)

        xclicks = xtemp
        yclicks = ytemp
        
        xclicks = [0]+xclicks+[subim.shape[1]-1]
        yclicks = [yclicks[0]]+yclicks+[yclicks[-1]]

        xclicks = np.array(xclicks)
        yclicks = np.array(yclicks)
        
        idx = np.argsort(xclicks)
        
        xclicks = xclicks[idx]
        yclicks = yclicks[idx]
        
        x = np.arange(subim.shape[1])
        interpolator = interp1d(xclicks,yclicks,kind='cubic')

        isos_position = interpolator(x)+zmin

        
        plt.imshow(np.log(im),cmap='gray',interpolation='none')
        plt.autoscale(False)
        plt.plot(x,isos_position)
        plt.plot(xclicks,yclicks+zmin,'rx')
        plt.savefig('%s_marked.png'%tag,dpi=300)
        np.savetxt('%s_isos_depth.txt'%tag,isos_position)
        plt.close()
        
        xclicks,yclicks,junk = collector([subim],titles=['click edges of normal retina'])
        np.savetxt('%s_normal_edges.txt'%tag,np.round(np.sort(xclicks)))
        xclicks,yclicks,junk = collector([subim],titles=['click edges of druse'])
        np.savetxt('%s_druse_edges.txt'%tag,np.round(np.sort(xclicks)))
