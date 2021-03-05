"""
Performs the STIP algorithm to an IFG stack to select scatterers for use in time-series analysis. 
"""

#=================MODULES=========================

from datetime import datetime as dt
import numpy as np
#import re
#import statistics as st
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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

def main(w, phase):
    count, hhist, vhist = STIP(w, phase)
    
    hdfWrite(f"w{int(w)}d{int(N)}.hdf5", count, hhist, vhist)


    #hdfWrite(f"coh.hdf5", count, hhist, vhist)

    
#==================READING DATA====================

def hdfRead(file, header=False):
    """Short function to extract data from the file. Returns False if no data is found
    for that header. """
    with h5.File(file, 'r') as f:
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

def STIP(window, IFGs):
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
    
    with ProcessPoolExecutor() as executor:
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
    t = time.time()
    N, r, c = IFGs.shape
    N += 1
    lag = np.arange(-(N-2), N-1, 1)             # Define array of time lags
    print (N)

    zlagix = int(np.where(lag==0)[0])           # Define the index of 0 lag

    argarray = np.zeros((len(lag), r, c))*1j    # Prepare the argmax array
    radius = int((window-1)/2)
    lag2z = np.arange(-(N-2), 1, 1)
    for nix, n in enumerate(lag):               # Loop through the time lags
        
        pc, pn = prepareNeighbour(IFGs, n, radius, h, v)

        e = neCoherence(pc, pn)
        sume = ne.evaluate("sum(e, axis=0)")
        argarray[nix] = sume

        #if n != 0:
        #    argarray[-nix] = sume
        #else: 
        #    pass
        #print (f"Lag {nix} time: {time.time()-t} /n Loops {loops}")
    argmaxix = np.argmax(argarray, axis=0)      # Finding the index of the maximum ????????????????????
    
    siblingMask = (argmaxix == zlagix)          # Creating mask of px where argmax
                                                # is at zero time lag
    hArr, vArr, hvix = maskTransform(tran, siblingMask, h, v)
    print (time.time()-t)
    return siblingMask, hArr, vArr, hvix

def prepareNeighbour(arr, n, radius, h, v):
    """Prepare array of neighbours
    n: time lag"""

    N, r, c = arr.shape
    N += 1
    IFGix = np.arange(N-1)
    lags = np.arange(-(N-2), N-1)
    i = IFGix+n
    maskl = i >= min(IFGix)
    maskg = i <= max(IFGix)
    mask = maskl*maskg
    central_ix = IFGix[mask]
    neighbour_ix = IFGix[mask]+n

    central = arr[central_ix]

    neighbour = np.pad(arr[neighbour_ix],((0,0),(radius+1,radius+1),(radius+1,radius+1)),'reflect')[:, radius+v+1:-radius+v-1, radius+h+1:-radius+h-1]

    return central, neighbour


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

def neCoherence(arr1, arr2):
    """Much faster that coherence()"""
    return ne.evaluate("exp(1j*(arr1-arr2))")


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
