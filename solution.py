# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:59:53 2014

@author: dave
"""

import numpy as np
from support import e_r_corr
from support import topocent
from support import tropo



# This function uses the Gauss-Newton method to itertively solve the non-linear least squares 
# problem of calculating a receivers position based on range measurements between the receiver and a
# number of satellites.
def LeastSquareSatPos(satPos,pRanges):
    
  # GPS speed of light
  c = 2.99792458e8;
  # Define initial receiver position guess
  pos = np.zeros(4);
  # Count the number of active satellites
  numSats = np.size(pRanges,0);
  # Define some of the data structures we are going to be using in the iterative least squares
  A = np.zeros((numSats,4));
  #
  rotSatPos = satPos;
  
  for itrNdx in range(0,7):
    
    # Using the ECEF coordinate frame assumes that the earth's rotation is taken into account.  
    # The satellite positions passed into this function are referenced to the GPS time of 
    # transmission.  The receiver position calculation is done at the time of receiving. Hence the 
    # satellite and receiver positions are calculated at different times which translates to 
    # different coordinate systems.  To correct for the difference between the coordinate frames, 
    # the receiver time (and hence coordinate system) will be used as the reference frame.  The 
    # satellite position coordinate frame must be changed.  The lines below calculate the pulse 
    # travel time by estimating the distance between each satellite and the current estimated receiver
    # position.  This is updated each iteration because the estimate of the receiver position changes 
    # each iteration.
    rangeEst = np.sqrt( (satPos[0,:]-pos[0])**2 + (satPos[1,:]-pos[1])**2 + (satPos[2,:]-pos[2])**2 );
    travelTimes = rangeEst / c;
    
    # Using the travel times estimated above, calculate the rotation angle between the two ECEF 
    # coordinate frames and rotate the GPS time of transmission frame to the receiver time of 
    # reception frame.  Only do this on the last iteration (saves times and has very little overall
    # impact on accuracy)
    if itrNdx == 6:
      for satNdx in range(0,numSats):
        rotSatPos[:,satNdx] = e_r_corr(travelTimes[satNdx], satPos[:,satNdx]);
      # Recalculate the range estimates since the position of each satellite has been updated
      rangeEst = np.sqrt( (rotSatPos[0,:]-pos[0])**2 + (rotSatPos[1,:]-pos[1])**2 + (rotSatPos[2,:]-pos[2])**2 );
    else:
      # To speed up the function, skipp updating the satellite positions due to coordinate frame 
      # differences when not on the last iteration. The search for the global minima is minimally 
      # changed due to this.
      rotSatPos = satPos;
  
    # Calculate the Jacobian.  This is a critical step in the Gauss-Newton method and forms the "A"
    # or basis function matrix for each iter linear least squares operation.
    A = JacobianCalc(pos,rotSatPos,rangeEst);
  
    # Calculate the delta-pseudoranges.  This is based on the current estimated receiver position, 
    # the satellites in the proper "receiver" coordinate-frame, and the range error due to the 
    # receivers time error.
    pRangeDelta = pRanges - rangeEst - pos[3];
    
    # Perform a least squares iteration of our linearized system
    x = np.linalg.lstsq(A,pRangeDelta)[0];
    
    # Update the position estimate with the incremental update
    pos = pos + x;
  #   residual = norm( [pos(1)-satPos(satNdx,1), pos(2)-satPos(satNdx,2), pos(3)-satPos(satNdx,3) ] ) - pRanges(satNdx)
  #   + pos(4)*c
  return pos;



def JacobianCalc(pos1,pos2,rangeEst):

  jacob = np.zeros((np.size(pos2,1),4));
  jacob[:,0] = (pos1[0] - pos2[0,:])/rangeEst;
  jacob[:,1] = (pos1[1] - pos2[1,:])/rangeEst;
  jacob[:,2] = (pos1[2] - pos2[2,:])/rangeEst;
  
  # Insert the jacobian into the "A" matrix.  The fourth variable (the ones) represent the 
  # receiver time error.
  jacob[:,3] = np.ones(np.size(pos2,1));
  
  return jacob;

