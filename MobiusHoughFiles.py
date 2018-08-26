import numpy as np
from numpy.random import randint
import pandas as pd
pd.set_option("display.max_colwidth", 1000000)
from math import sin
from math import cos
from numpy import arctan2
from numpy import sqrt 
import json

#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d
#import seaborn as sns

#####################################stuff added by JRS ########################
#matplotlib stuff to make plots
#import matplotlib
#import matplotlib.pyplot as plt
#fig, first = plt.subplots()
#This job requires a file index number to be passed
import sys
import errno
#print "Number of Arguments %i " % len(sys.argv)
#print "Jobname %s" % sys.argv[1] 
path_to_train = sys.argv[1]

index         = int(sys.argv[2])
print("Index of File to read in Train %i = " % index)
# use test file: 
#index = 0

LoopTheta     = int(sys.argv[3])
print("File List Directory %s = " % path_to_train)
print("File Index to run %i   = " % index )
print("LoopTheta Value   %i   = " % LoopTheta)
import glob
#path_to_train = '/Users/johnrsmith/Kaggle/Datafiles/train_100_events/'
filepath  = path_to_train + '/*-hits.csv'
print("Path to Files %s" % filepath)
filelist = []
files = glob.glob(filepath)
for file in files:
    strip1 = file.replace('hits.csv','')
    strip  = strip1.replace(path_to_train,'')
    print(strip)
    filelist.append(strip)

print("Length of Filelist %i" % len(filelist))
print("The file Number to run is %i" % index)
print("The file to read from is %s" % filelist[index])
path_to_hits  = path_to_train+filelist[index]+'hits.csv'
path_to_truth = path_to_train+filelist[index]+'truth.csv'
###############################################################################

'''

This section is for the main global search parameters.  Something that needs to be optimized

'''
#theta is the angle from positive x-axis to the center of the helix circle
nThetas = 500   # nThetas+1 is the number of bins in theta 

#curv is the curvature of the helix cross-section
maxCurv = .005
nCurv = 40      # number of bins in the Curvature of the helix "C"   r =1/(2C)

#phi is the angle of the line formed by s = arc-length and the z-coordinate
nPhi = 500      # number of bins in the polar angle

#C is the closest distance the unwound straight line in (s,z) coordinates comes to the origin
maxC = .00005
# Chris used 5 
#nC = 5
nC = 6          # distance to the origin. 

#make the search lists

#thetaList  = np.linspace(0,     2*np.pi, nThetas+1)
thetaList  = np.linspace(0,     np.pi, nThetas+1)
thetaList
#phiList    = np.linspace(0,     2*np.pi, nPhi+1)
phiList    = np.linspace(0,     np.pi, nPhi+1)
curvList   = np.linspace(0,     maxCurv, nCurv+1)
CList      = np.linspace(-maxC, maxC,    nC+1)

deltaTheta = thetaList[1] - thetaList[0]
deltaPhi   = phiList[1]   - phiList[0]
deltaCurv  = curvList[1]  - curvList[0]
deltaC     = CList[1]     - CList[0]

'''

This section is reading data from the file. It must be changed to each computer to read the right file

'''
def getHitsWithParticleId():
    """
        returns hits from a random event along with an extra column designating the 'particle_id' as a pandas data frame
        it also returns the set of particle_ids
    """
    event         = "%03d" % randint(0, 100)
    event = "031"
    #path_to_train = "D:/Kaggle/ParticleTrack/train_100_events/"
    #path_to_train = "/home/szjrsmit/Kaggle/Datafiles/train_100_events/"
    #path_to_train = "/Users/johnrsmith/Kaggle/Datafiles/train_100_events/"
    #event_prefix  = "event000001"+event+"-"
    #event_prefix  = filelist[index]
    #path_to_train = "/Users/johnrsmith/Kaggle/Datafiles/test/"
    #event_prefix  = "event000000"+event+"-"
    #hits          = pd.read_csv(path_to_train+event_prefix+"hits.csv")
    hits          = pd.read_csv(path_to_hits)
    hits          = hits[['hit_id', 'x', 'y', 'z']]
    #print("event: "+event+" with "+str(hits.shape[0])+ " hits" )
    #truth         = pd.read_csv(path_to_train+event_prefix+"truth.csv",converters={'particle_id':str})
    truth         = pd.read_csv(path_to_truth,converters={'particle_id':str})
    truth         = truth[['hit_id', 'particle_id', 'tpx', 'tpy', 'tpz']]
    combined      = pd.merge(hits, truth, on='hit_id')
    ids           = truth.particle_id.unique()
    return combined, ids

'''

This section is some basic functions. Each will extend a dataframe to a new column.
They require that other columns have already been added, so they have to be done in some natural order.

'''

def getR(myHits):
    myHits['r']    = myHits['arc'] * myHits['arc'] + myHits['z'] * myHits['z']


def getC(myHits, phi):
    myHits['C']    = myHits['arc'] / myHits['r'] * sin(phi) + myHits['z'] / myHits['r'] * cos(phi)


def getCurv(myHits, theta):
    myHits['curv'] = (2 * myHits['x'] * cos(theta) + 2 * myHits['y'] * sin(theta)) / ( myHits['x'] ** 2 + myHits['y'] ** 2)


def getArcLength(myHits, theta):
    myHits['arc'] = arctan2(myHits['x'] * sin(theta) - myHits['y'] * cos(theta),
                            -myHits['x'] * cos(theta) - myHits['y'] * sin(theta) + 1 / myHits['curv']) / myHits['curv']

def getTruePt(myHits):
    myHits['TruePt']    = np.sqrt(myHits['tpx']*myHits['tpx'] + myHits['tpy']*myHits['tpy'])


'''

This section has the main Hough Transforms.
'''

#'''
#def firstHough(myHits):
#    """
#            Takes a list of hits and finds the best circle through them that passes through the origin
#            The circle is determined by the angle to the center and the curvature of the circle, as in Mobius coordinates
#    """
#    maxCount  = 0
#    bestTheta = -1
#    bestCurv  = -1
#    for theta in thetaList:
#        getCurv(myHits, theta)
#        #first.plot(myHits, theta)
#        for curv in curvList:
#            count = myHits[np.abs(myHits['curv'] - curv) < deltaCurv].shape[0]
#            if count > maxCount:
#                maxCount  = count
#                bestTheta = theta
#                bestCurv  = curv
#
#    print(maxCount)
#    return maxCount, bestTheta, bestCurv
#'''

def firstHough(myHits):
    """
            Takes a list of hits and finds the best circle through them that passes through the origin
            The circle is determined by the angle to the center and the curvature of the circle, as in Mobius coordinates
    """
    maxCount  = 0
    bestTheta = -1
    bestCurv  = -1
    nCell     = 0
    iTheta    = 0
    print(" LoopTheta Value = %i " % LoopTheta)
    for theta in thetaList:
        iTheta    += 1
        if iTheta != LoopTheta: nCell += nCurv;

        if iTheta == LoopTheta:
            getCurv(myHits, theta)
            #first.plot(myHits, theta)
            for curv in curvList:
                # the new cell logic
                hitList = []
                for index, row in myHits.iterrows():
                    #print(" here is a row ", row)
                    if np.abs(row['curv'] - curv) < deltaCurv:
                        #hitList = row
                        keyList   = list(row.keys())
                        itemList  = list(row.items())
                        #print(" Here is the itemList = ", itemList)
                        #print(" Row Items = ", row.items())
                        #valDict   = dict(row.values())
                        #valDict   = {row.values()}
                        #valDict    = {k: v for (k,v) in row.values()}
                        #print(" Here is the valDict = ", valDict)
                        hitList.append(itemList)
                        #hitList.append(index) # append to 3rd item which is an empty list
                        #hitList.append(row) # append to 3rd item which is an empty list
                        hitString = json.dumps(hitList)

                cell = [{'nCell' : nCell, 'theta' : theta, 'curv' : curv, 'listOfHits' : hitString}]
                frame = pd.DataFrame(cell, columns=['nCell', 'theta', 'curv', 'listOfHits'])
                frameName = "pandasCell" + str(nCell) + ".csv"
                frame.to_csv(frameName)
                #readInFrame = pd.read_csv(frameName)
                #readInnCell = readInFrame['nCell']
                #readInTheta = readInFrame['theta']
                #readIncurv  = readInFrame['curv']
                #readInHits  = readInFrame['listOfHits'].to_string(index=False,header=False)
                #print("Printing readInHits for Mike = %s" % readInHits)
                #hitsAsList  = json.loads(readInHits)
                # use the hits dataframe to read in the relevant hit, i.e. myHits.loc[hitsAsList[0]] 
                # will return the row out of the hit data file
                # cellContainer.append(cell)
                count = myHits[np.abs(myHits['curv'] - curv) < deltaCurv].shape[0]
                nCell += 1
                if count > maxCount:
                    maxCount = count
                    bestTheta = theta
                    bestCurv = curv

                print("Cell Number = %i " % nCell)
                print(maxCount)
    return maxCount, bestTheta, bestCurv

'''
def firstHough(myHits):
        """
                Takes a list of hits and finds the best circle through them that passes through the origin
                The circle is determined by the angle to the center and the curvature of the circle, as in Mobius coordinates
        """
        maxCount  = 0
        bestTheta = -1
        bestCurv  = -1
        nCell     = 0
        iTheta    = 0
        #cellContainer = []
        for theta in thetaList:
                iTheta    += 1
                if iTheta != LoopTheta: 
                        nCell += nCurv;
                else 
                        getCurv(myHits, theta)
                        #first.plot(myHits, theta)
                        for curv in curvList:
                                #########################################################
                                # the new cell logic
                                cell = [nCell, theta, curv, []]
                                nhits = 0
                                for index, row in myHits.iterrows():
                                        if np.abs(row['curv'] - curv) < deltaCurv:
                                                nhits = nhits + 1
                                                cell[3].append(row) # append to 3rd item which is an empty list
                                
                                print("Cell Number %i  Length of Cell =  %i", nCell,  nhits) 
                                frame = pd.DataFrame(cell, index=['nCell', 'theta', 'curv', 'listOfHits'])
                                frameName = "pandasCell" + str(nCell) + ".csv"
                                frame.to_csv(frameName)
                                #cellContainer.append(cell)
                                ########################################################
                                ## the new cell logic
                                #hitList = []
                                #for index, row in myHits.iterrows():
                                #       if np.abs(row['curv'] - curv) < deltaCurv:
                                #               hitList.append(index) # append to 3rd item which is an empty list
                                #               hitString = json.dumps(hitList)
                                #
                                #cell = [{'nCell' : nCell, 'theta' : theta, 'curv' : curv, 'listOfHits' : hitString}]
                                #frame = pd.DataFrame(cell, columns=['nCell', 'theta', 'curv', 'listOfHits'])
                                #frameName = "pandasCell" + str(nCell) + ".csv"
                                #frame.to_csv(frameName)
                                #readInFrame = pd.read_csv(frameName)
                                #readInnCell = readInFrame['nCell']
                                #readInTheta = readInFrame['theta']
                                #readIncurv = readInFrame['curv']
                                #readInHits = readInFrame['listOfHits'].to_string(index=False,header=False)
                                #hitsAsList = json.loads(readInHits)
                                ## use the hits dataframe to read in the relevant hit, i.e. myHits.loc[hitsAsList[0]] 
                                ## will return the row out of the hit data file
                                ## cellContainer.append(cell)
                                ########################################################
                                count = myHits[np.abs(myHits['curv'] - curv) < deltaCurv].shape[0]
                                nCell += 1
                                if count > maxCount:
                                        maxCount = count
                                        bestTheta = theta
                                        bestCurv = curv


        print("Number of Cells = %i " % nCell)
        #print "Length of Cell Container = %i " % lent(cellContainer) 
        print(maxCount)
        return maxCount, bestTheta, bestCurv
'''

def secondHough(myHits):
    """
                Takes a list of hits and does a second Hough transform to find the best line to represent the unwound helix
    """
    maxCount  = 0
    bestPhi   = -1
    bestC     = -1
    getR(myHits)
    for phi in phiList:
        getC(myHits, phi)
        for C in CList:
            count = myHits[np.abs(myHits['C'] - C) < deltaC].shape[0]
            if count > maxCount:
                maxCount = count
                bestPhi  = phi
                bestC    = C
    return maxCount, bestPhi, bestC


'''

Here is the main routine

'''


def main():
    """
        Gets a random track from a random event.
        Does both Hough transforms and reports after each one,
        which cell is best,
        how many total hits are in the best cell
        how many of the hits in the track are left in the best cell
    """
    hits, trackIds = getHitsWithParticleId()
    id             = np.random.choice(trackIds)
    # for the purpose of checking we will freeze the track in file 12
    #id = '612497933098549248'
    #id  = '49546192970842112'
    print("Track ID number = ", id)
    trackHits      = hits.loc[hits['particle_id'] == id].copy()
    print("chosen track has " + str(trackHits.shape[0]) + " hits")
    Pmom = getTruePt(trackHits)
    print(' Momentum ', trackHits['TruePt'])

    if trackHits.shape[0]<3:
        print("Track does not have enough hits to be meaningful")
        print("Try again")
        return

    print("doing first Hough...")
    #fig, first = plt.subplots()
    #maxCount, bestTheta, bestCurv = firstHough(trackHits)
    maxCount, bestTheta, bestCurv = firstHough(hits)
    #first.plot(theta, curv, 'o')
    #plt.show()
    print("maxCount: "  + str(maxCount))
    print("bestTheta: " + str(bestTheta))
    print("bestCurv: "  + str(bestCurv))
    getCurv(hits, bestTheta)
    getCurv(trackHits, bestTheta)
    trackHitsInCell = trackHits.loc[np.abs(trackHits['curv'] - bestCurv) < deltaCurv].copy()
    HitsInCell      = hits.loc[np.abs(hits['curv'] - bestCurv) < deltaCurv].copy()
    nCellTrackHits  = trackHitsInCell.shape[0]
    nCellHits       = HitsInCell.shape[0]
    getArcLength(HitsInCell, bestTheta)
    getArcLength(trackHitsInCell, bestTheta)
    print("total of all hits in this cell: " + str(nCellHits))
    print("total track hits in this cell: " + str(nCellTrackHits) + " of " + str(trackHits.shape[0]))
    print("")

    print("doing second Hough...")
    maxCount, bestPhi, bestC = secondHough(trackHitsInCell)
    print("maxCount: " + str(maxCount))
    print("bestPhi: "  + str(bestPhi))
    print("bestC: "    + str(bestC))
    getR(HitsInCell)
    getR(trackHitsInCell)
    getC(HitsInCell, bestPhi)
    getC(trackHitsInCell, bestPhi)
    trackHitsInSecondCell = trackHitsInCell.loc[np.abs(trackHitsInCell['C'] - bestC) < deltaC].copy()
    HitsInSecondCell      = HitsInCell.loc[np.abs(HitsInCell['C'] - bestC) < deltaC].copy()
    nTrackHitsInSecond    = trackHitsInSecondCell.shape[0]
    nHitsInSecond         = HitsInSecondCell.shape[0]
    print("total of all hits in second cell: " + str(nHitsInSecond))
    print("total track hits in second cell: " + str(nTrackHitsInSecond) + " of " + str(trackHits.shape[0]))

if __name__ == "__main__":
    main()
