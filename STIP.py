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
import numexpr as ne
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

print ("\nModules loaded\n")

#====================FILES=========================

fn = r'/home/jacob/InSAR_workspace/data/doncaster/vel_jacob_doncaster.h5'
fn2 = r'/home/jacob/InSAR_workspace/data/doncaster/data_jacob_doncaster.h5'

#fn = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/vel_jacob_doncaster.h5'
#fn2 = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/data_jacob_doncaster.h5'

#fn = "C:/Users/jcobc/Documents/University/doncaster/vel_jacob_doncaster.h5"
#fn2 = "C:/Users/jcobc/Documents/University/doncaster/data_jacob_doncaster.h5"
#==================PARAMETERS======================

N = 18
w = 25

#=====================CODE=========================

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(N, w):
    #phase_noise = extractData(fn2, 'Phase')
    #normPhase = normalisedPhase(phase[-20:], amp[-20:], [800, 407], [936, 377])
    count, hhist, vhist = STIP(N, w, phase[81:81+N])
    
    hdfWrite(f"w{int(w)}d{int(N)}_reflect.hdf5", count, hhist, vhist)
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

def STIP(N, window, IFGs):
    """Implimentation of STIP
    N = no. of SLCs
    N-1 = no. of IFGs"""
    
    N = len(IFGs)
    
    t1 = time.time()
    #dates = np.asarray(extractData('Date', f))

    d, r, c = IFGs.shape # For dates, rows, columns

    siblingCount = np.zeros((r, c))
    dlist = neighbourhood(window)
    radius = int((window-1)/2)
    windowMask = circleMask(radius)

    tran = [dlist[i] for i in range(len(dlist))\
            if windowMask[i] == True]                           # List of transformations
    
    hhistory = np.empty((len(tran), r, c))
    vhistory = np.empty((len(tran), r, c))
    
    print (tran)
    
    arguments = [[IFGs, *hv, window, tran] for hv in tran]
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = [executor.submit(siblingTest, *ar) for ar in arguments]
        
        for f in as_completed(results):
            siblingMask, hArr, vArr, hvix = f.result()
            hhistory[hvix] = hArr
            vhistory[hvix] = vArr
            siblingCount += siblingMask*1
    print (time.time() - t1)
    return siblingCount, hhistory, vhistory
    
def siblingTest(IFGs, h, v, window, tran):
    
    print (f'h={h}, v={v}')
    
    N, r, c = IFGs.shape
    lag = np.arange(-(N-2), N-1, 1)             # Define array of time lags
    zlagix = int(np.where(lag==0)[0])           # Define the index of 0 lag
    argarray = np.zeros((len(lag), r, c))*1j    # Prepare the argmax array
    radius = int((window-1)/2)
    for nix, n in enumerate(lag):               # Loop through the time lags
        esum = np.zeros((r, c))*1j              # Prepare the coherence sum array
        
        for m in np.arange(N):                  # Loop through the IFGs
            padMask = np.zeros((r, c))          # Prepare mask (for outside pixels)
            if 1 <= m+n <= N-2:                 # Verify that this time lag is possible
                pc = IFGs[m]                    # Array of central pixel  
                  
                # Array of neighbour pixels (pc but shifted across)
                pn = np.pad(IFGs[m+n],radius+1,'reflect')[radius+v+1:\
                      -radius+v-1, radius+h+1:-radius+h-1]
                      
                # Populate padMask
                padMask = np.pad(padMask,radius+1,'constant',\
                          constant_values=1)[radius+v+1:-radius+v-1,\
                          radius+h+1:-radius+h-1].astype('bool')
            
                e = neCoherence(pc, pn)         # Calc coherence
                e[padMask] = 0                  # Set "outside" values to 0 coh
                esum += e                       # Add to sum
            else:
                pass
        
        argarray[nix] = esum                    # Populating the argmax array
        
    argmaxix = np.argmax(argarray, axis=0)      # Finding the index of the maximum
    
    siblingMask = (argmaxix == zlagix)          # Creating mask of px where argmax
                                                # is at zero time lag
    
    hArr, vArr, hvix = maskTransform(tran, siblingMask, h, v)
    
    return siblingMask, hArr, vArr, hvix

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
    """
    Returns two arrays of the h and v transformation from that pixel to the
    sibling pixel. Also returns the index of the transformation out of the 
    list of transformations. 

    These transfromations allow matrix multiplication.

    """

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

def neCoherence(arr1, arr2):
    """Much faster that coherence()"""
    e = math.e
    return ne.evaluate("e**(1j*arr1) * e**(-1j*arr2)")


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
        
        normed = phaseWrap(normed)
        
        pArrOut[i] = normed
    
    return pArrOut 
    
def phaseWrap(arr):
    
    grtpiMask = arr > np.pi
    lsrpiMask = arr < -np.pi

    arr[grtpiMask] = arr[grtpiMask] - 2*np.pi
    arr[lsrpiMask] = arr[lsrpiMask] + 2*np.pi
    
    return arr
    
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
        pltData = phaseWrap(np.asarray(pltData))
        ax[0].plot(pltData, 'b.', alpha=0.5)
        ax[1].plot(d, 'b.', alpha=0.5)
    ax[0].plot(centralPhase-centralPhase, 'rx')#, label='Central pixel')
    ax[1].plot(centralPhase, 'rx')

    legend_elements = [Line2D( [0], [0], color='blue', lw=0, marker='.', label='STIP Neighbours'),
                       Line2D( [0], [0], color='red', lw=0, marker='x', label='Central pixel')]
    ax[0].legend(handles=legend_elements, bbox_to_anchor=(1.01, 1.0), loc='upper left')

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



