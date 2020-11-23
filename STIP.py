"""
Performs the STIP algorithm to an IFG stack to select scatterers for use in
time-series analysis. 
"""

import sys
import getopt
import os
import shutil
import subprocess as subp
# import sqlite3
import datetime as dt
import numpy as np
import re

print ("\nModules loaded\n")

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv == None:
        argv = sys.argv

    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:", ["help"])
        except getopt.getOptError as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print (__doc__)
                return 0
            elif o == '-i':
                ifgdir = a
            elif o == '-o':
                outdir = a

		# Check if the directories exist. 
        if not os.path.exists(ifgdir):
            raise Usage(f'Input data directory {ifgdir} does not exist.')
        elif not os.path.isdir(ifgdir):
            raise Usage(f'Given Input data directory {ifgdir} is not a directory.')

        if not os.path.exists(outdir):
            raise Usage(f'Input data directory {outdir} does not exist.')
        elif not os.path.isdir(outdir):
            raise Usage(f'Given Input data directory {outdir} is not a directory.')
        else:
            print ("Output dir OK\n")

    # No directories were specified.
    except Usage as err:
        print (sys.stderr, "\nWoops, something went wrong:")
        print (sys.stderr, "  "+str(err.msg))
        print (sys.stderr, "\nFor help, use -h or --help.\n")
        return 2

    # Creates a list of IFGs to be analysed. 
    ifglist = list(os.listdir(ifgdir))
    ifgs = "\n".join(ifglist)
    print (f"The following IFGs have been found: \n{ifgs}")
    print ("\n==============================================================\n")

    load_ifgs(ifglist)

def load_ifgs(ifglist):
    """
    Loading the IFGs for processing. 
    """
    print (ifglist)
    date_pairs = []
    if ifglist:
        procslavelist = list()
        for ifg in ifglist:
            mtch = re.search('(\d+)[-_\s]+(\d+)',ifg)
            procslavelist += mtch.groups()
            print (procslavelist)
            date_pair = [dt.datetime.strptime(ds,'%y%m%d') for ds in mtch.groups()]
            date_pairs.append(date_pair)
#         with open(ifgListFile) as f:
#             for line in f:
#                 mtch = re.search('(\d+)[-_\s]+(\d+)',line)
#                 procslavelist += mtch.groups()
#                 date_pair = [dt.datetime.strptime(ds,'%Y%m%d') for ds in mtch.groups()]
#                 date_pairs.append(date_pair)

main()
