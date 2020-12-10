"""
Performs the STIP algorithm to an IFG stack to select scatterers for use in
time-series analysis. 
"""

#=================MODULES=========================

import sys
import getopt
import os
import shutil
import subprocess as subp
# import sqlite3
from datetime import datetime as dt
import numpy as np
import re
import h5py as h5
import statistics as st
import matplotlib.pyplot as plt

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

def main():
    f = open_hdf(fn2)
    dates = ext_data('Date', f)
    phase = ext_data('Phase', f)
    lon = ext_data('Longitude', f)
    lat = ext_data('Latitude', f)

    
#==================READING DATA====================
        
def open_hdf(fn):
    """Open the file and assign vars to datasets. """ 
    global f
    f = h5.File(fn, 'r')
    headers = list(f.keys())    
    return f

def ext_data(header, hdfo):
    """Short function to extract data from the file. Returns False if no data is found
    for that header. """

    try:
        data = hdfo[header]
    except KeyError:
        print (f'No dataset for {header} found. Set to False. \n')
        data = False
    return data
    
def read_csv(fname):
    return np.genfromtxt(fname, delimiter=',')
    
#=================WRITING DATA=====================

def write_csv(name, arr):
    np.savetxt(name, arr, delimiter=',')
    return
    
def write_hdf(name, *dsets):
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
    dates = np.asarray(ext_data('Date', f))
    ifgs = ifgs[-(N-1):]
    datesn, r, c = ifgs.shape # For dates, rows, columns

    STIP_count = np.zeros((r, c))
    dlist = neighbourhood(window)
    print (dlist)
    nhistory = np.empty((len(dlist), r, c), dtype=object)
    cmpx = 1j
    for h, v in dlist:
        argarray = []
         
        print (f'h={h}, v={v}')
        # Looping through the padding to create the neighbour matrices
        lag = np.arange(-(N-2), N-1, 1) 
        zlagix = int(np.where(lag==0)[0])
        
        for n in lag:
        # Looping through the argmax
            esum = np.zeros((r, c))*1j
            for m in np.arange(len(ifgs)):
            # Looping through the dates 
                
                if 1 <= m+n <= N-2:
                    
                    # Center pixel
                    pc = ifgs[m]
                    # Neighbour pixel (same matrix but shifted some way based on h & v.
                    pn = padding(ifgs[m+n], h, v)
                    e = coherence(pc, pn)
                    # Exponential sum
                    esum += e
                else:
                    pass
                
            # Add to array to perform argmax on
            argarray.append(esum)
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
        
        hvmask, historyix = maskTransform(dlist, STIP_mask, h, v)
        nhistory[historyix] = hvmask
        
        STIP_count += STIP_mask*1

        # The loop is restarted for the next h and v values.

    return STIP_count, nhistory

def maskTransform(dlist, mask, h, v):
    
    ix = dlist.index((h, v))    
    
    r, c = mask.shape
    maskf = mask.flatten()
    l = np.empty((r, c), dtype=object).flatten()#[]
    t = (h, v)
    
    for i, p in enumerate(maskf):
        if p:
            l[i] = t
            # l.append(t)
        else:
            l[i] = np.nan
            # l.append(np.nan)
    
    arrOut = np.asarray(l).reshape((r, c))

    return arrOut, ix

def coherence(arr1, arr2):
    return np.exp(1j*arr1)*np.exp(-1j*arr2)

def padding(arr, h, v):
    """If h -ve then add columns to the left. +ve add columns to the right
    IF v -ve then add rows to the top. +ve add rows to the bottom."""
    
    r,c = arr.shape
    
    arr_new=arr.copy()
    
    if h < 0:
        # Adding columns to left
        to_add = np.zeros((r, np.abs(h)))
        arr_new = np.concatenate((to_add, arr_new[:,:h]), axis=1)
        
    elif h > 0: 
        # Adding columns to right
        to_add = np.zeros((r, h))
        arr_new = np.concatenate((arr_new[:,h:], to_add), axis=1)
    else:
        pass
        
    if v < 0:
        # Adding rows to the top
        to_add = np.zeros((np.abs(v), c))
        arr_new = np.concatenate((to_add, arr_new[:v,:]), axis=0)
    elif v > 0:
        # Adding rows to the bottom
        to_add = np.zeros((v, c))
        arr_new = np.concatenate((arr_new[v:,:], to_add), axis=0)
    else:
        pass
        
    return arr_new

def neighbourhood(windowDim):
    """setting up the neighbourhoods (number of rows/cols for padding)"""
    
    d = int((windowDim-1)/2)
    drange = np.arange(-d, d+1, 1)
    dlist = [(i, j) for i in drange for j in drange if (i, j)!=(0, 0)]

    
    return dlist

def cropData(arr):
    n=60
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
    
#===================PLOTTING FUNCTIONS=====================

def radCoor(arr):
    r, c = arr.shape

    x = np.asarray(([i for i in range(c)]*r)).reshape((r, c))
    y = np.asarray(([[i]*c for i in range(r)]))
    if not mask:
        plt.scatter(x, y, c=arr, s=0.5)
    plt.show()
    return ""

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

        ax.plot(datesf, series, '-')

    plt.show()
            
    return

def hist(array, nbins):
    try:
        a, b = array.shape
        arrayf = array.flatten()
    except:
        print ('Array was already 1D - proceeding')
        arrayf = array.copy()
    fig, ax = plt.subplots()

    ax.hist(arrayf, nbins, rwidth=0.95)
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



# main()

f = open_hdf(fn2)

phase = cropData(np.asarray(ext_data('Phase', f)))
lon = cropData(np.asarray(ext_data('Longitude', f)))
lat = cropData(np.asarray(ext_data('Latitude', f)))

count, nhistory = STIP(N, w, phase)

write_hdf("w11d18.hdf5", count, nhistory)


#np.savetxt(f'STIP_{int(w)}by{int(w)}_{int(N)}dates_comp.csv', count, delimiter=',')

# scatter(lon, lat, count)


