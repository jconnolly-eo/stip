"""
Performs the STIP algorithm to an IFG stack to select scatterers for use in
time-series analysis. 
"""

#=================MODULES=========================

#import sys
#import getopt
#import os
#import shutil
#import subprocess as subp
# import sqlite3
from datetime import datetime as dt
import numpy as np
import re
import h5py as h5
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import random
import cmath


print ("\nModules loaded\n")

#====================FILES=========================

fn = r'/home/jacob/InSAR_workspace/data/doncaster/vel_jacob_doncaster.h5'
fn2 = r'/home/jacob/InSAR_workspace/data/doncaster/data_jacob_doncaster.h5'

#fn = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/vel_jacob_doncaster.h5'
#fn2 = r'/nfs/a1/insar/sentinel1/UK/jacob_doncaster/data_jacob_doncaster.h5'

#==================PARAMETERS======================

N = 18
w = 11

#=====================CODE=========================

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(N, w):
    #phase_noise = extractData(fn2, 'Phase')
    normPhase = normalisedPhase(phase[-20:], amp[-20:], (918, 63), (1020, 77))
    count, hhist, vhist = STIP(N, w, normPhase)
    
    hdfWrite(f"w{int(w)}d{int(N)}_norm.hdf5", count, hhist, vhist)
    #hdfWrite(f"coh.hdf5", count, hhist, vhist)

    
#==================READING DATA====================
        
#def open_hdf(fn):
#    """Open the file and assign vars to datasets. """ 
#    global f
#    f = h5.File(fn, 'r')
#    headers = list(f.keys())    
#    return f

def extractData(file, header):
    """Short function to extract data from the file. Returns False if no data is found
    for that header. """
    with h5.File(file) as f:
        try:
            data = np.asarray(f[header])
            
        except KeyError:
            print (f'No dataset for {header} found. Set to False. \n')
            headers = list(f.keys())
            print ('Possible headers: \n', headers)
            data = False
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
    print (dlist)
    hhistory = np.empty((len(dlist), r, c))
    vhistory = np.empty((len(dlist), r, c))
    cmpx = 1j
    for h, v in dlist:
        # argarray = []
        
         
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
        
        hmask, vmask, historyix = maskTransform(dlist, STIP_mask, h, v)
        hhistory[historyix] = hmask
        vhistory[historyix] = vmask
        
        STIP_count += STIP_mask*1

        # The loop is restarted for the next h and v values.
    t2 = time.time()
    print (t2-t1)
    return STIP_count, hhistory, vhistory


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
    
    #pArrOut = np.zeros(dim)
    pArrOut = phase.copy()
    
    pRegion = phase[0, dim[1]-br[1]:dim[1]-tl[1], tl[0]:br[0]]
    aRegion = amp[0, dim[1]-br[1]:dim[1]-tl[1], tl[0]:br[0]]
    complexRegion = toComplex(pRegion, aRegion)
    sumRegion = np.sum(complexRegion)
    normed = np.exp(1j*phase)*np.exp(1j*cmath.phase(sumRegion.conjugate()))
    
    pArrOut = np.arctan(normed.imag/normed.real)
    
    #print (tl[0], br[0], tl[1], br[1])
    #for i in range(dim[0]):
    
        #pRegion = phase[i, dim[1]-br[1]:dim[1]-tl[1], tl[0]:br[0]]
        #aRegion = amp[i, dim[1]-br[1]:dim[1]-tl[1], tl[0]:br[0]]
        #print (pRegion.shape, aRegion.shape)
        #complexRegion = toComplex(pRegion, aRegion) #np.exp(1j*pRegion)*aRegion
        
        #sumRegion = np.sum(complexRegion)
        #print (sumRegion)
        
        #normed = np.exp(1j*phase[i])*sumRegion.conjugate()
        
        #pArrOut[i] = np.arctan(normed.imag/normed.real)
        
        #pArrOut[i] = np.asarray([cmath.phase(p) for p in (np.exp(1j*phase[i])*sumRegion.conjugate()).flatten()]).reshape(phase[i].shape)
        #pArrOut[i] = np.asarray([cmath.phase(])
        #pReturn = cmath.phase(sumRegion)
        #print (pReturn)
        
        #pArr = phase[i]pReturn
        
        #pArrOut[i] = pArr
    
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
    plt.show()
    

def time_series(data, dateix, dates, mask):
    fig, ax = plt.subplots()
    print ('Created subplot \n')

    datebool = type(dateix) is tuple
    if datebool:
        a, b = dateix
        data = data[a:b]
        dates = dates[a:b]
    else:
        data = data[-dateix:]
        dates = dates[-dateix:]

    
    datesf = np.asarray([dt.strptime(str(d[0]), '%Y%m%d') for d in dates]) 
    print ('Formatted dates \n')
    x, y = mask.shape
    mask = mask.flatten()
    print (f'Data shape: {data.shape}')
    datat = data.T.reshape((x*y, len(data)))[mask]
    print ('Reshaped data \n')
    # Plotting
    print (f'Plotting {len(datat)}')
    for series in datat:

        ax.plot(datesf, series, '.')

    plt.show()
            
    return

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

def contour(arr):
    fig, ax = plt.subplots()
    contf = ax.contourf(arr)
    cbar = fig.colorbar(contf)
    plt.savefig('STIP_cont.png')
    plt.show()
    
def scatter(lon, lat, col):
    fig, ax = plt.subplots()
    sca = ax.scatter(lon, lat, c=col, s=1)
    cbar = fig.colorbar(sca)
    plt.savefig('STIP_scatter.png')
    plt.show()

def cumulative(count):
    c = []
    for i in range(np.max(count)):
        mask = (count > i)
        c.append(np.sum(mask*1))
        
    plt.plot(c)
    plt.show()
    
    
def plotNeighbours(x,y, hhist, vhist, phase):
    """Plot the neighbours"""
    data=[]
    mask = ~np.isnan(hhist[:, y, x])
    print (mask)
    coords = tuple(zip(hhist[:, y, x][mask], vhist[:, y, x][mask]))
    print (coords)
    fig, ax = plt.subplots()
    for h, v in coords:
        series = phase[:, int(y+v), int(x+h)]
        data.append(series)
    print (len(data))
    centralPhase = phase[:, int(y), int(x)]
    centralAmp = amp[:, int(y), int(x)]
    for d in data:
        ax.plot(d-centralPhase, 'b.')
    ax.plot(centralPhase-centralPhase, 'r.')
    plt.show()

def plotNeighboursComp(x,y, hhist, vhist, phase, amp):
    """Plot the neighbours"""

    #Lists to add the phase and amp data to 
    dataPhase=[]
    dataAmp=[]
    
    # Create a mask to remove the pixels with no STIPs
    mask = ~np.isnan(hhist[:, y, x])
    
    # Create a list of tuples with the coordinates of the neighbouring
    # STIP pixels.
    coords = tuple(zip(hhist[:, y, x][mask], vhist[:, y, x][mask]))

    # Append the data for the neighbours to the lists
    for h, v in coords:
        phaseSeries = phase[:, int(y+v), int(x+h)]
        ampSeries = amp[:, int(y+v), int(x+h)]
        dataPhase.append(phaseSeries)
        dataAmp.append(ampSeries)
    
    # Calculate the complex version of the central pixel
    centralPhase = phase[:, int(y), int(x)]
    centralAmp = amp[:, int(y), int(x)]
    centralComp = toComplex(centralPhase, centralAmp)
    
    print (centralComp)
    # Create the axes
    fig, ax = plt.subplots()
    
    # Plot each of the 
    for p, a in zip(dataPhase, dataAmp):
        comp = toComplex(p, a) - centralComp
        print (comp)
        normalised = np.asarray([cmath.phase(c) for c in comp])
        ax.plot(normalised, 'b.')
    ax.plot(centralPhase-centralPhase, 'r.')
    plt.show()
    
def toComplex(p, a):
    return np.exp(1j*p)*a
    
def av_comp(arr):
    """    """
    i = 0+1j
    comp = np.exp(i*arr)
    
    avphase = cmath.phase(np.mean(comp))
    
    return avphase 
    
def indexCount(count, num):
    l = zip(*np.where(count==num))
    
    

#main(N, w)
phase = cropData(extractData(fn2, 'Phase'))
amp = cropData(extractData(fn2, 'Amplitude'))

#count = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_0'))
#hhistory = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_1'))
#vhistory = cropData(extractData('w11-varied-dates/w11d18.hdf5', 'data_2'))
#phase = (np.asarray(extractData(fn2, 'Phase')))
#d, r, c = phase.shape
#phase_noise = np.asarray(np.random.random((d, r, c)))*2*np.pi - np.pi



