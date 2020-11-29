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
import datetime as dt
import numpy as np
import re
import h5py as h5
import statistics as st

print ("\nModules loaded\n")

#====================FILES=========================

fn = r'/home/jacob/InSAR_workspace/data/doncaster/vel_jacob_doncaster.h5'
fn2 = r'/home/jacob/InSAR_workspace/data/doncaster/data_jacob_doncaster.h5'

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

def time_series(data, lat, lon, ix=False):
    """To plot a time series from the data"""
    
    fig, ax = plt.subplots()        
    plt.show()
    
    return

def coh(arr):
    """Find values in arrays that are not nan and then create an array of indices 
    corresponding to these coherent pixels. """
    n, m = arr.shape
    mask = ~np.isnan(np.asarray(arr))
    ix = np.arange(n*m).reshape(n, m)
    cix = ix[mask]
    
    return mask, cix

def STIP(N, ifgs):
    """Implimentation of STIP
    N = no. of SLCs
    N-1 = no. of IFGs"""
    STIPix = []
    d, r, c = ifgs.shape # For dates, rows, columns
#    ix = np.arange(r*c).reshape(r, c)
    tall = []
    for n in np.arange(-(N-2),N-1,1):
        
        for m in np.arange(N):
            ppad = np.pad(ifgs, 10)
            tsum = np.zeros((r, c-1))
            pc = np.delete(ifgs[m], -1, axis=1)
            pn = np.delete(ifgs[m+n], 0, axis=1)
            
            t = np.exp(pc)*np.exp(-pn)
            tsum += t
            
        tall.append(tsum)
    
    argmaxix = np.argmax(tall, axis=0)
    STIP_mask = (argmaxix == st.median(range(len(tall))))

    return tall, STIP_mask
    
def STIP(N, ifgs):
    """Implimentation of STIP
    N = no. of SLCs
    N-1 = no. of IFGs"""
    STIPix = []
    d, r, c = ifgs.shape # For dates, rows, columns
#    ix = np.arange(r*c).reshape(r, c)
    tall = []
    for n in np.arange(-(N-2),N-1,1):
        
        for m in np.arange(N):
            
            ppad = np.pad(ifgs, 10)
            tsum = np.zeros((r, c-1))
            pc = np.delete(ifgs[m], -1, axis=1)
            pn = np.delete(ifgs[m+n], 0, axis=1)
            
            t = np.exp(pc)*np.exp(-pn)
            tsum += t
            
        tall.append(tsum)
    
    argmaxix = np.argmax(tall, axis=0)
    STIP_mask = (argmaxix == st.median(range(len(tall))))

    return tall, STIP_mask
    
def padding(arr, hv=(0,0)):
    """If h -ve then add columns to the left. +ve add columns to the right
    IF v -ve then add rows to the top. +ve add rows to the right."""
    
    h, v = hv
    
    r,c = arr.shape
    
    arr_new=arr
    
    if h < 0:
        # Adding columns to left
        to_add = np.zeros((r, np.abs(h)))
        arr_new = np.concatenate((to_add, arr[:,:h]), axis=1)
        
    elif h > 0: 
        # Adding columns to right
        to_add = np.zeros((r, h))
        arr_new = np.concatenate((arr[:,h:], to_add[:,h:]), axis=1)
    else:
        arr_new = arr
        
    if v < 0:
        # Adding rows to the top
        to_add = np.zeros((np.abs(v), c))
        arr_new = np.concatenate((to_add, arr_new[:v,:]), axis=0)
    elif v > 0:
        # Adding rows to the bottom
        to_add = np.zeros((v, c))
        arr_new = np.concatenate((arr_new[v:,:], to_add), axis=0)
    else:
        arr_new = arr
        
    return arr_new

def neighbourhood(n, matrix):
    """setting up the neighbourhoods"""
    pad_matrix = np.pad(matrix, n)
    r,c = matrix.shape
    
    
    return
#    for n in np.arange(-(N-2),N-1,1):
#        for i in np.arange(r*c):
#            x, y = np.where(ix == i)
#            t_all = []
#            
#            for m in np.arange(N):
#                try:
#                    pc = ifgs[m,x,y] # center pixel
#                    pn = ifgs[m+n,x+1,y] # neighbour pixel
#                except IndexError:
#                    pass
#                t = np.exp(pc)*np.exp(-pn)# For test
#                
#            tsum = sum(t_all)
#            print (tsum)

def cityblock(n):
    center = np.pad(np.arange(n+1), (n,0), 'reflect')
    cb = np.zeros((2*n+1,2*n+1))
    for i,x in enumerate(center):
        cb[i,:] = center+x
        
    cb = np.where(cb > n, False, cb)
    
    truth = np.where(cb > 0, True, cb)
    
    truth_coords = np.where(truth==1)
    
    return cb, truth, truth_coords

main()

f = open_hdf(fn2)
phase = np.asarray(ext_data('Phase', f))
