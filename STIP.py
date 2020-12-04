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

N = 18
w = 15

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

    
#===================================================
        
def open_hdf(fn):
    """Open the file and assign vars to datasets. """ 
    global f
    f = h5.File(fn, 'r')
    headers = list(f.keys())
#    vel = ext_data('Velocity', f)
#    phase = ext_data('Phase', f)

#    lon = ext_data('Longitude', f)
#    lat = ext_data('Latitude', f)
#    dates = ext_data('Date', f)
    
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
    cmpx = 1j
    for h in dlist:
        
        for v in dlist:
            argarray = []
            
            print (f'Starting with h={h}, v={v}')
            # Looping through the padding to create the neighbour matrices
            
            for n in np.arange(-(N-2),N-1,1):
            # Looping through the argmax
                for m in np.arange(len(ifgs)):
                # Looping through the dates 
                    esum = np.zeros((r, c))*1j
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
            print (np.asarray(argarray).shape)
            argmaxix = np.argmax(argarray, axis=0)
        
            # If the index of the argmax for a specific value is in the middle of the stack
            # (meaning zero time lag) then it is a STIP. This adds 1 to that pixel if it is
            # otherwise adds zero.
            # STIP_count is therefore a 2d array (with same dimensions as ifgs) where the 
            # number corresponds to the number of STIP pixels.
            STIP_count += (argmaxix == st.median(range(len(argarray))))*1
        
        # The loop is restarted for the next h and v values.

    return STIP_count


def coherence(arr1, arr2):
    return np.exp(1j*arr1)*np.exp(-1j*arr2)

def padding(arr, h, v):
    """If h -ve then add columns to the left. +ve add columns to the right
    IF v -ve then add rows to the top. +ve add rows to the right."""
    
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
    
    d = round(int(windowDim)/2)
    
    dlist = np.arange(-d, d+1, 1)
    dlist = dlist[np.abs(dlist)>0]
    
    return dlist

def cityblock(n):
    center = np.pad(np.arange(n+1), (n,0), 'reflect')
    cb = np.zeros((2*n+1,2*n+1))
    for i,x in enumerate(center):
        cb[i,:] = center+x
        
    cb = np.where(cb > n, False, cb)
    
    truth = np.where(cb > 0, True, cb)
    
    truth_coords = np.where(truth==1)
    
    return cb, truth, truth_coords
    
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

    ax.hist(arrayf, nbins)
    plt.show()

def read_csv(fname):
    return np.genfromtxt(fname, delimiter=',')

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

phase = np.asarray(ext_data('Phase', f))
lon = np.asarray(ext_data('Longitude', f))
lat = np.asarray(ext_data('Latitude', f))

#count = STIP(N, w, phase)

#np.savetxt(f'STIP_{int(w)}by{int(w)}_{int(N)}dates_comp.csv', count, delimiter=',')

# scatter(lon, lat, count)


