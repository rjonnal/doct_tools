from matplotlib import pyplot as plt
from clicky import collector
import numpy as np
import sys,os
from scipy.misc import imread
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


class Summary:

    def __init__(self,fn):
        partial = os.path.split(fn)[1] # split the filename into a list [directory,filename] and get the second
        tokens = partial.split('_') # tokenize the filename using underscores
        pupil_position = int(tokens[-2])
        
        tag = os.path.splitext(fn)[0]
        isos_depth = np.loadtxt('%s_isos_depth.txt'%tag).astype(int)
        normal_edges = np.loadtxt('%s_normal_edges.txt'%tag).astype(int)
        druse_edges = np.loadtxt('%s_druse_edges.txt'%tag).astype(int)

        druse_width = np.diff(sorted(druse_edges))
        region_width = druse_width//5

        normal_edges = sorted(normal_edges.astype(int))
        normal_start,normal_end = normal_edges
        
        druse_left_start=np.min(druse_edges)
        druse_left_end=druse_left_start+region_width
        druse_center_start=np.mean(druse_edges).astype(int)-region_width//2
        druse_center_end=druse_center_start+region_width
        druse_right_start=np.max(druse_edges)-region_width
        druse_right_end=druse_right_start+region_width


        colors = 'rgb'
        im = imread(fn)

        inner_retina = []
        for x in range(im.shape[1]):
            inner = im[isos_depth[x]-40:isos_depth[x]-5,x]
            inner_retina.append(inner)

        inner_retina = np.array(inner_retina).mean(axis=1)
        im = im/inner_retina

        druse_left = []
        druse_right = []
        druse_center = []
        normal = []

        temp = []
        for x in range(druse_left_start,druse_left_end+1):
            druse_left.append(im[isos_depth[x],x])
        druse_left_mean = np.mean(druse_left)
        temp = []
        for x in range(druse_center_start,druse_center_end+1):
            druse_center.append(im[isos_depth[x],x])
        druse_center_mean = np.mean(druse_center)
        temp = []
        for x in range(druse_right_start,druse_right_end+1):
            druse_right.append(im[isos_depth[x],x])
        druse_right_mean = np.mean(druse_right)
        temp = []
        for x in range(normal_start,normal_end+1):
            normal.append(im[isos_depth[x],x])
        normal_mean = np.mean(normal)
        
        plt.imshow(np.log(im),cmap='gray',interpolation='none')
        druse_x1s = [druse_left_start,druse_center_start,druse_right_start]
        druse_x2s = [druse_left_end,druse_center_end,druse_right_end]
        for idx,(x1,x2) in enumerate(zip(druse_x1s,druse_x2s)):
            plt.axvspan(x1,x2,color=colors[idx],alpha=0.25)

        plt.axvspan(normal_start,normal_end,color='y',alpha=0.25)
        plt.savefig('%s_regions_marked.png'%tag,dpi=300)

        print '%d\t%0.3f\t%0.3f\t%0.3f\t%0.3f'%(pupil_position,normal_mean,druse_left_mean,druse_center_mean,druse_right_mean)
        

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
