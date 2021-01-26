"""
Performs the STIP algorithm to an IFG stack to select scatterers for use in time-series analysis. 
"""

#=================MODULES=========================

from datetime import datetime as dt
import numpy as np
import re
import h5py as h5
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import time
import random
import cmath

print ("\nModules loaded\n")

#====================FILES=========================

#fn = r'/home/jacob/InSAR_workspace/data/doncaster/vel_jacob_doncaster.h5'
#fn2 = r'/home/jacob/InSAR_workspace/data/doncaster/data_jacob_doncaster.h5'

fn = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/vel_jacob_doncaster.h5'
fn2 = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/data_jacob_doncaster.h5'

#fn = "C:/Users/jcobc/Documents/University/doncaster/vel_jacob_doncaster.h5"
#fn2 = "C:/Users/jcobc/Documents/University/doncaster/data_jacob_doncaster.h5"
#==================PARAMETERS======================

N = 40
w = 25

#=====================CODE=========================

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(N, w):
    #phase_noise = extractData(fn2, 'Phase')
    #normPhase = normalisedPhase(phase[-20:], amp[-20:], [800, 407], [936, 377])
    count, hhist, vhist = STIP(N, w, phase[81:81+N])
    
    hdfWrite(f"w{int(w)}d{int(N)}_norm.hdf5", count, hhist, vhist)
    #hdfWrite(f"coh.hdf5", count, hhist, vhist)

    
#==================READING DATA====================

def hdfRead(file, header=False):
    """Short function to extract data from the file. Returns False if no data is found
    for that header. """
    with h5.File(file) as f:
        if header:
            try:
                data = np.asarray(f[header])

            except KeyError:
                print (f'No dataset for {header} found. Set to False. \n')
                headers = list(f.keys())
                print ('Possible headers: \n', headers)
                data = False
        else:
            data = [np.asarray(f[str(h)]) for h in list(f.keys())]
    return data
    
def csvRead(fname):
    return np.genfromtxt(fname, delimiter=',')
    
#=================WRITING DATA=====================

def csvWrite(name, arr):
    np.savetxt(name, arr, delimiter=',')
    return
    
def hdfWrite(name, *dsets):
    print (f'Writing to {name}.\n')
    
    f = h5.File(name, "w")
    for ix, dset in enumerate(dsets):
        shape = dset.shape
        dsetname = f"data_{int(ix)}"
        d = f.create_dataset(dsetname, shape, data=dset)
    f.close()
    statement = f"Data saved at {name}"
    return statement

#================MAIN FUNCTIONS====================

def STIP(N, window, ifgs):
    """Implimentation of STIP
    N = no. of SLCs
    N-1 = no. of IFGs"""
    t1 = time.time()
#    dates = np.asarray(extractData('Date', f))
    ifgs = ifgs[-(N-1):]
    datesn, r, c = ifgs.shape # For dates, rows, columns

    STIP_count = np.zeros((r, c))
    dlist = neighbourhood(window)
    windowMask = circleMask(int((window-1)/2))
    # print (dlist)

    cmpx = 1j
    maskeddlist = [dlist[i] for i in range(len(dlist)) if windowMask[i] == True]
    hhistory = np.empty((len(maskeddlist), r, c))
    vhistory = np.empty((len(maskeddlist), r, c))
    print (maskeddlist)
    for h, v in maskeddlist:
        print (f'h={h}, v={v}')
        # Looping through the padding to create the neighbour matrices
        lag = np.arange(-(N-2), N-1, 1) 
        zlagix = int(np.where(lag==0)[0])
        argarray = np.zeros((len(lag), r, c))*cmpx
        for nix, n in enumerate(lag):
            #print (f'n={n}')
        # Looping through the argmax
            esum = np.zeros((r, c))*1j
            for m in np.arange(len(ifgs)):
            # Looping through the dates 
                
                if 1 <= m+n <= N-2:
                    #print (f'm={m}')
                    # Center pixel
                    pc = ifgs[m]
                    # Neighbour pixel (same matrix but shifted some way based on h & v.
                    pn = padding(ifgs[m+n], h, v)
                    padmask = (pn==0)
                    e = coherence(pc, pn)
                    e[padmask] = 0
                    # Exponential sum
                    esum += e
                else:
                    pass
                
            # Add to array to perform argmax on
            # argarray.append(esum)
            argarray[nix] = esum
        # Looks at all the sums for a specific pixel across all the time lags and finds
        # the index of the max value
        # print (np.asarray(argarray).shape)
        argmaxix = np.argmax(argarray, axis=0)
        
        # If the index of the argmax for a specific value is in the middle of the stack
        # (meaning zero time lag) then it is a STIP. This adds 1 to that pixel if it is
        # otherwise adds zero.
        # STIP_count is therefore a 2d array (with same dimensions as ifgs) where the 
        # number corresponds to the number of STIP pixels.
        STIP_mask = (argmaxix == zlagix) #st.median(range(len(argarray))))
        
        hmask, vmask, historyix = maskTransform(maskeddlist, STIP_mask, h, v)
        hhistory[historyix] = hmask
        vhistory[historyix] = vmask
        
        STIP_count += STIP_mask*1

        # The loop is restarted for the next h and v values.
    t2 = time.time()
    print (t2-t1)
    return STIP_count, hhistory, vhistory
    
def circleMask(radius):
    """
    Creates a mask of the transformations so that the window is geometrically correct. 
    """
    squareTransform = neighbourhood(radius*2 + 1)
    
    # Create square matrix for with mask for circle
    #mask = np.ones((radius, radius))
    Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
    dist_from_center = np.sqrt((X)**2 + (Y*3.966)**2)
    mask = dist_from_center <= radius
    
    maskFlatten = [mask[d[1]+radius, d[0]+radius] for d in squareTransform]
    
    return maskFlatten
    
def maskTransform(dlist, mask, h, v):
    
    ix = dlist.index((h, v))    
    
    r, c = mask.shape
    maskf = mask.flatten()
    ha = np.empty((r, c)).flatten()
    va = np.empty((r, c)).flatten()
    t = (h, v)
    
    for i, p in enumerate(maskf):
        if p:
            ha[i] = h
            va[i] = v
            # l.append(t)
        else:
            ha[i], va[i] = np.nan, np.nan
            # l.append(np.nan)
    arrhOut = np.asarray(ha).reshape((r, c))
    arrvOut = np.asarray(va).reshape((r, c))

    return arrhOut, arrvOut,  ix

def coherence(arr1, arr2):
    return np.exp(1j*arr1)*np.exp(-1j*arr2)

def padding(arr, h, v):
    """If h -ve then add columns to the left. +ve add columns to the right
    IF v -ve then add rows to the top. +ve add rows to the bottom."""
    
    r,c = arr.shape
    
    arr_new=arr.copy()
    
    if h < 0:
        # Adding columns to left
        to_add = np.zeros((r, np.abs(h)))# - np.inf
        arr_new = np.concatenate((to_add, arr_new[:,:h]), axis=1)
        
    elif h > 0: 
        # Adding columns to right
        to_add = np.zeros((r, h))# - np.inf
        arr_new = np.concatenate((arr_new[:,h:], to_add), axis=1)
    else:
        pass
        
    if v < 0:
        # Adding rows to the top
        to_add = np.zeros((np.abs(v), c))# - np.inf
        arr_new = np.concatenate((to_add, arr_new[:v,:]), axis=0)
    elif v > 0:
        # Adding rows to the bottom
        to_add = np.zeros((v, c))# - np.inf
        arr_new = np.concatenate((arr_new[v:,:], to_add), axis=0)
    else:
        pass
        
    return arr_new
    
def padReflection(arr, h, v):

    r, c = arr.shape
    
    arr_new=arr.copy()
    
    tl = h < 0 and v < 0
    bl = h < 0 and v > 0
    tr = h > 0 and v < 0 
    br = h > 0 and v > 0
    
    if h < 0:
        left = arr.T[::-1].T[:,h:]
    elif h > 0:
        right = arr.T[::-1].T[:,:h]
    else: 
        pass
        
    if v < 0:
        top = arr[::-1][v:, :]
    elif v > 0:
        bottom = arr[::-1][:v, :]
    else:
        pass
        
    if tl:
        pass
    elif bl:
        pass
    elif tr:
        pass
    elif br: 
        pass
        

def neighbourhood(windowDim):
    """setting up the neighbourhoods (number of rows/cols for padding)"""
    
    d = int((windowDim-1)/2)
    drange = np.arange(-d, d+1, 1)
    dlist = [(i, j) for i in drange for j in drange if (i, j)!=(0, 0)]

    
    return dlist

def cropData(arr):
    n=79
    if len(list(arr.shape))==3:
        cropped = arr[:, :, :-n]
    elif len(list(arr.shape))==2:
        cropped = arr[:,:-n]
    return cropped

def cityblock(n):
    center = np.pad(np.arange(n+1), (n,0), 'reflect')
    cb = np.zeros((2*n+1,2*n+1))
    for i,x in enumerate(center):
        cb[i,:] = center+x
        
    cb = np.where(cb > n, False, cb)
    
    truth = np.where(cb > 0, True, cb)
    
    truth_coords = np.where(truth==1)
    
    return cb, truth, truth_coords

    
def normalise(arr):
    """To normalise an array so that is sums to one."""
    arrc = np.asarray(np.asarray(arr).copy())
    maxs = []
    for k in arrc:
        mask = ~np.isnan(k)
        if len(k[mask]) > 0:
            maxs.append(max(k[mask]))
    maxn = max(maxs)
    n, m = arr.shape
    for i in range(n):
        arrc[i] = arrc[i]/maxn
    return arrc, maxn
    
def normalisedPhase(phase, amp, tl, br):
    """Function to normalise the complex phase of an IFG based 
    on a region given by the coordinates tl (top-left) and br (bottom-right)."""
    dim = phase.shape
    
    pArrOut = phase.copy()
    c = np.dstack((tl, br))[0]
    
    for i in range(dim[0]):
    
        pRegion = phase[i, c[1, 1]:c[1, 0], c[0, 0]:c[0, 1]]
        aRegion = amp[i, c[1, 1]:c[1, 0], c[0, 0]:c[0, 1]]

        sumRegion = np.sum(toComplex(pRegion, aRegion))

        normed = phase[i] - cmath.phase(sumRegion)

        grtpiMask = normed > np.pi
        lsrpiMask = normed < -np.pi

        normed[grtpiMask] = normed[grtpiMask] - 2*np.pi
        normed[lsrpiMask] = normed[lsrpiMask] + 2*np.pi
        
        #normed = np.exp(1j*phase[i])  *  np.exp(1j*cmath.phase(sumRegion)).conjugate()
        
        pArrOut[i] = normed#np.arctan(normed.imag/normed.real)
    
    return pArrOut 

    
#===================PLOTTING FUNCTIONS=====================

def radCoor(arr, colour=True, mask=True, region=[0, 0, 0, 0]):
    r, c = arr.shape

    x = np.asarray(([i for i in range(c)]*r)).reshape((r, c))
    y = np.asarray(([[i]*c for i in np.arange(r)]))#, 0, -1)]))
    fig, ax = plt.subplots()
    plt.axis([np.nanmin(x), np.nanmax(x), np.nanmax(y), np.nanmin(y)])
    if colour:
        p = ax.scatter(x[mask], y[mask], c=arr[mask], s=0.5)
        fig.colorbar(p, ax=ax)
    else:
        ax.scatter(x[mask], y[mask], s=0.5)
    
    ax.add_patch(Rectangle((region[0], region[3]), region[2]-region[0], region[1]-region[3],
                 edgecolor='red',
                 fill=False,
                 lw=1))
    plt.axis([np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)])
    #fig.set_figheight(6)
    #fig.set_figwidth(6*1.1297)
    ax.set_aspect(1.0/ax.get_data_ratio()*0.8852)#1.1297)
    plt.show()

def hist(array):
    try:
        a, b = array.shape
        arrayf = array.flatten()
    except:
        print ('Array was already 1D - proceeding')
        arrayf = array.copy()
    fig, ax = plt.subplots()
    bins = np.arange(0, 110, 5) - 2.5
    ax.hist(arrayf, bins, rwidth=0.95)
    plt.show()
    
def scatter(lon, lat, col):
    fig, ax = plt.subplots()
    sca = ax.scatter(lon, lat, c=col, s=1)
    cbar = fig.colorbar(sca)
    #plt.savefig('STIP_scatter.png')
    ax.set_aspect(1.0/ax.get_data_ratio()*0.8852)
    plt.show()
   
def plotNeighbours(x,y, hhist, vhist, phase):
    """Not used..."""
    data=[]
    mask = ~np.isnan(hhist[:, y, x])
    print (mask)
    coords = np.dstack((hhist[:, y, x][mask], vhist[:, y, x][mask]))[0]
    print (coords)
    fig, ax = plt.subplots(2)
    for c in coords:
        h, v = c
        series = phase[:, int(y+v), int(x+h)]
        data.append(series)
    print (len(data))
    centralPhase = phase[:, int(y), int(x)]
#     centralAmp = amp[:, int(y), int(x)]
    for d in data:
        pltData = d-centralPhase
        for i, p in enumerate(pltData):
            if p>np.pi:
                pltData[i] = p - 2*np.pi
            elif p<-np.pi:
                pltData[i] = p + 2*np.pi
        ax[0].plot(pltData, 'b.', alpha=0.5)
        ax[1].plot(d, 'b.', alpha=0.5)
    ax[0].plot(centralPhase-centralPhase, 'rx')#, label='Central pixel')
    ax[1].plot(centralPhase, 'rx')
    #ax[0].legend()
    #ax[1].legend()
    legend_elements = [Line2D( [0], [0], color='blue', lw=0, marker='.', label='STIP Neighbours'),
                       Line2D( [0], [0], color='red', lw=0, marker='x', label='Central pixel')]
    ax[0].legend(handles=legend_elements, bbox_to_anchor=(1.01, 1.0), loc='upper left')
    #ax[1].legend(handles=legend_elements)
    plt.show()
    
def plotNeighbourRad(x, y, hhist, vhist, count):
    """Plot position of neighbours in radar coords"""
    
    data = np.zeros((count.shape))
    
    mask = ~np.isnan(hhist[:, y, x])
    print (mask)
    coords = np.dstack((hhist[:, y, x][mask], vhist[:, y, x][mask]))[0]
    
    xpx = [x+c[0] for c in coords]
    ypx = [y+c[1] for c in coords]
    
    fig, ax = plt.subplots()
    r, c = count.shape
    mx = np.asarray(([i for i in range(c)]*r)).reshape((r, c))
    my = np.asarray(([[i]*c for i in np.arange(r)]))
    p = ax.scatter(mx, my, c=count, s=0.5)
    
    ax.plot(x, y, 'rx')
    ax.plot(xpx, ypx, 'b.')
    ax.set_aspect(1.0/ax.get_data_ratio()*0.8852)
    plt.show()
    
def toComplex(p, a):
    return np.exp(1j*p)*a
    
#main(N, w)
#phase = cropData(hdfRead(fn2, 'Phase'))
#amp = cropData(hdfRead(fn2, 'Amplitude'))

#count = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_0'))
#hhistory = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_1'))
#vhistory = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_2'))
#phase = (np.asarray(extractData(fn2, 'Phase')))
#d, r, c = phase.shape
#phase_noise = np.asarray(np.random.random((d, r, c)))*2*np.pi - np.pi



