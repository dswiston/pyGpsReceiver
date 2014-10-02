# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:05:02 2014

@author: dave
"""

import numpy as np

def CalcSatPos(satData,transTime):

  satPos = np.zeros(3);

  # Define constants
  mu = 3.986005e14;                 # meters^3/sec^2 - WGS-84 value of the Earth's universal gravitational parameter
  omegaE = 7.2921151467e-5;         # rad/sec − WGS-84 value of the Earth's rotation rate
  piVal = 3.1415926535898;
  F = -4.442807633e-10; # Constant, [sec/(meter)^(1/2)]
  
  #--- Find time difference ---------------------------------------------
  dt = transTime - satData.Toc;
  dt = CheckTimeWrap(dt);
  
  #--- Calculate clock correction ---------------------------------------
  clkCorr = satData.af2 * dt**2 + satData.af1 * dt + satData.af0 - satData.Tgd;
  time = transTime - clkCorr;
  
  
  ## Find satellite's position ----------------------------------------------

  # Restore semi-major axis
  a = satData.sqrtA**2;

  # Time correction
  tk = time - satData.Toe;
  tk = CheckTimeWrap(tk);

  # Initial mean motion
  n0 = np.sqrt(mu / np.power(a,3));
  # Mean motion
  n = n0 + satData.deltaN;

  # Mean anomaly
  M = satData.M0 + n * tk;
  # Reduce mean anomaly to between 0 and 360 deg
  M = np.remainder(M + 2*piVal, 2*piVal);

  #Initial guess of eccentric anomaly
  E = M;

  #--- Iteratively compute eccentric anomaly ----------------------------
  for ndx in range(0,10):
    E_old = E;
    E = M + satData.e * np.sin(E);
    dE = np.remainder(E - E_old, 2*piVal);

    if (abs(dE) < 1.e-12):
      # Necessary precision is reached, exit from the loop
      break;

  #Reduce eccentric anomaly to between 0 and 360 deg
  E = np.remainder(E + 2*piVal, 2*piVal);

  #Compute relativistic correction term
  dtr = F * satData.e * satData.sqrtA * np.sin(E);

  #Calculate the true anomaly
  nu = np.arctan2(np.sqrt(1 - satData.e**2) * np.sin(E), np.cos(E) - satData.e);

  #Compute angle phi
  phi = nu + satData.omegaSmall;   # <--------------------------------------- NOT SURE
  #Reduce phi to between 0 and 360 deg
  phi = np.remainder(phi, 2*piVal);

  #Correct argument of latitude
  u = phi + satData.Cuc * np.cos(2*phi) + satData.Cus * np.sin(2*phi);
  #Correct radius
  r = a * (1 - satData.e*np.cos(E)) + satData.Crc * np.cos(2*phi) + satData.Crs * np.sin(2*phi);
  #Correct inclination
  i = satData.i0 + satData.IDOT * tk + satData.Cic * np.cos(2*phi) + satData.Cis * np.sin(2*phi);

  #Compute the angle between the ascending node and the Greenwich meridian
  Omega = satData.omegaBig + (satData.omegaDot - omegaE)*tk - omegaE * satData.Toe;
  #Reduce to between 0 and 360 deg
  Omega = np.remainder(Omega + 2*piVal, 2*piVal);

  #--- Compute satellite coordinates ------------------------------------
  satPos[0] = np.cos(u)*r * np.cos(Omega) - np.sin(u)*r * np.cos(i)*np.sin(Omega);
  satPos[1] = np.cos(u)*r * np.sin(Omega) + np.sin(u)*r * np.cos(i)*np.cos(Omega);
  satPos[2] = np.sin(u)*r * np.sin(i);


  # Include relativistic correction in clock correction --------------------
  clkCorr = satData.af2 * dt**2 + satData.af1 * dt + satData.af0 - satData.Tgd + dtr;
                     
  return (satPos, clkCorr)
                       
                       
def CheckTimeWrap(time):

  halfWeek = 302400;     # seconds
  if (time > halfWeek):
    time = time - 2*halfWeek;
  elif (time < -halfWeek):
    time = time + 2*halfWeek;
  
  return time
  
  
  
  
  
  
def CalculatePseudoRange(trackTime,fs):

  # Define the speed of light
  c = 2.99792458e8;
  
  # Find number of samples per bit
  samplesPerCode = fs/1.023e6 * 1023;
  
  # Compute fraction of a millisecond timing
  time = trackTime / samplesPerCode;
  time2 = time - np.floor(min(time)) + 68.802000000000007;
  
  
  # Convert time to a distance
  psuedoRange = time2 * (c / 1000);
  
  return psuedoRange
  
  
  
  
  
  
  
  
def e_r_corr(traveltime, X_sat):

  Omegae_dot = np.array(7.292115147e-5);           #  rad/sec
  
  #--- Find rotation angle --------------------------------------------------
  omegatau   = Omegae_dot * traveltime;
  
  #--- Make a rotation matrix -----------------------------------------------
  R3 = [[np.cos(omegatau), np.sin(omegatau),0],[-np.sin(omegatau),np.cos(omegatau),0],[0,0,1]];
  
  #--- Do the rotation ------------------------------------------------------
  X_sat_rot = np.dot(R3,X_sat);
  
  return X_sat_rot;
  
  
  
  
  
  
  
  
  
def topocent(X, dx):

  dtr = np.pi/180;
  
  (phi, lam, h) = togeod(6378137, 298.257223563, X[0], X[1], X[2]);
  
  cl  = np.cos(lam * dtr);
  sl  = np.sin(lam * dtr);
  cb  = np.cos(phi * dtr); 
  sb  = np.sin(phi * dtr);
  
  F   = np.array([[-sl,-sb*cl,cb*cl],[cl,-sb*sl,cb*sl],[0,cb,sb]]);
  
  local_vector = np.dot(F.T,dx);
  E   = local_vector[0];
  N   = local_vector[1];
  U   = local_vector[2];
  
  hor_dis = np.sqrt(E**2 + N**2);
  
  if (hor_dis < 1.e-20):
      Az = 0;
      El = 90;
  else:
      Az = np.arctan2(E, N)/dtr;
      El = np.arctan2(U, hor_dis)/dtr;
  
  if (Az < 0):
      Az = Az + 360;
  
  D   = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2);

  return (Az, El, D);
  
  
  
  
  
  
  
  
  
  
def togeod(a, finv, X, Y, Z):
  
  h       = 0;
  tolsq   = 1.e-10;
  maxit   = 10;
  
  # compute radians-to-degree factor
  rtd     = 180/np.pi;
  
  # compute square of eccentricity
  if (finv < 1.e-20):
      esq = 0;
  else:
      esq = (2 - 1/finv) / finv;
  
  oneesq  = 1 - esq;
  
  # first guess
  # P is distance from spin axis
  P = np.sqrt(X**2+Y**2);
  # direct calculation of longitude
  
  if (P > 1.e-20):
      dlambda = np.arctan2(Y,X) * rtd;
  else:
      dlambda = 0;
  
  if (dlambda < 0):
      dlambda = dlambda + 360;
  
  # r is distance from origin (0,0,0)
  r = np.sqrt(P**2 + Z**2);
  
  if (r > 1.e-20):
      sinphi = Z/r;
  else:
      sinphi = 0;
  
  dphi = np.arcsin(sinphi);
  
  # initial value of height  =  distance from origin minus
  # approximate distance from origin to surface of ellipsoid
  if (r < 1.e-20):
      h = 0;
      return (dphi, dlambda, h);
  
  h = r - a*(1-sinphi*sinphi/finv);
  
  # iterate
  for i in range(0,maxit):
      sinphi  = np.sin(dphi);
      cosphi  = np.cos(dphi);
      
      # compute radius of curvature in prime vertical direction
      N_phi   = a/np.sqrt(1-esq*sinphi*sinphi);
      
      # compute residuals in P and Z
      dP      = P - (N_phi + h) * cosphi;
      dZ      = Z - (N_phi*oneesq + h) * sinphi;
      
      # update height and latitude
      h       = h + (sinphi*dZ + cosphi*dP);
      dphi    = dphi + (cosphi*dZ - sinphi*dP)/(N_phi + h);
      
      # test for convergence
      if (dP*dP + dZ*dZ < tolsq):
          break;
  
      # Not Converged--Warn user
      if (i == maxit):
          print("Problem in TOGEOD, did not converge in %s iterations" % (i));
  
  dphi = dphi * rtd;

  return (dphi, dlambda, h);
  
  
  
  
  
  
  
  
  
def tropo(sinel, hsta, p, tkel, hum, hp, htkel, hhum):
  
  a_e    = 6378.137;     # semi-major axis of earth ellipsoid
  b0     = 7.839257e-5;
  tlapse = -6.5;
  tkhum  = tkel + tlapse*(hhum-htkel);
  atkel  = 7.5*(tkhum-273.15) / (237.3+tkhum-273.15);
  e0     = 0.0611 * hum * np.power(10,atkel);
  tksea  = tkel - tlapse*htkel;
  em     = -978.77 / (2.8704e6*tlapse*1.0e-5);
  tkelh  = tksea + tlapse*hhum;
  e0sea  = e0 * np.power((tksea/tkelh),(4*em));
  tkelp  = tksea + tlapse*hp;
  psea   = p * np.power((tksea/tkelp),em);
  
  if (sinel < 0):
      sinel = 0;
  
  tropo   = 0;
  done    = 0;
  refsea  = 77.624e-6 / tksea;
  htop    = 1.1385e-5 / refsea;
  refsea  = refsea * psea;
  ref     = refsea * np.power(((htop-hsta)/htop),4);
  
  while 1:
      rtop = (a_e+htop)**2 - (a_e+hsta)**2*(1-sinel**2);
      
      # check to see if geometry is crazy
      if (rtop < 0):
          rtop = 0;  
      
      rtop = np.sqrt(rtop) - (a_e+hsta)*sinel;
      tmpA    = -sinel/(htop-hsta);
      tmpB    = -b0*(1-sinel**2) / (htop-hsta);
      rn   = np.zeros(8);
  
      for i in range(0,8):
          rn[i] = np.power(rtop,(i+2));
      
      alpha = [2*tmpA, 2*tmpA**2+4*tmpB/3, tmpA*(tmpA**2+3*tmpB), np.power(tmpA,4)/5+2.4*tmpA**2*tmpB+1.2*tmpB**2, 2*tmpA*tmpB*(tmpA**2+3*tmpB)/3, tmpB**2*(6*tmpA**2+4*tmpB)*1.428571e-1, 0, 0];
      
      if (tmpB**2 > 1.0e-35):
          alpha[6] = tmpA*np.power(tmpB,3)/2; 
          alpha[7] = np.power(tmpB,4)/9; 
  
      dr = rtop;
      dr = dr + np.dot(alpha,rn);
      tropo = tropo + dr*ref*1000;
      
      if (done):
          ddr = tropo; 
          break; 
      
      done    = 1;
      refsea  = (371900.0e-6/tksea-12.92e-6)/tksea;
      htop    = 1.1385e-5 * (1255/tksea+0.05)/refsea;
      ref     = refsea * e0sea * np.power((htop-hsta)/htop,4);
  
  return ddr;
  
  
  
def CaCodes(fs):

  # Each of the LFSRs are initialized with all ones
  g1 = np.ones(10);
  g2 = np.ones(10);
  
  # Create the "polynomial" mask for the g1 LFSR: 1 + x^3 + x^10
  g1Mask = np.array([0,0,1,0,0,0,0,0,0,1]);
  
  # Create the "polynomial" mask for the g2 LFSR: 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10
  g2Mask = np.array([0,1,1,0,0,1,0,1,1,1]);
  
  # Define the codes length - there are 1023 phase chips in the waveform
  codeLen = 1023;
  
  # Define the code phases - trim the last 5 (they are not to be used by satellites)
  codePhs = np.array([[2,6],[3,7],[4,8],[5,9],[1,9],[2,10],[1,8],[2,9],[3,10],[2,3],[3,4],[5,6], \
  [6,7],[7,8],[8,9],[9,10],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[1,3],[4,6],[5,7],[6,8],[7,9], \
  [8,10],[1,6],[2,7],[3,8],[4,9]]) - 1;
  
  # Define the code matrix
  code = np.zeros((codeLen,np.size(codePhs,0)));

  # Perform the LFSR and code generation operations
  for ndx in range(0,codeLen):
    
    code[ndx,:] = np.mod(g1[9] + np.sum(g2[codePhs],1),2);
    g1 = np.insert(g1[0:9], 0, np.mod(np.sum(g1 * g1Mask),2));
    g2 = np.insert(g2[0:9], 0, np.mod(np.sum(g2 * g2Mask),2));
      
  # If necessary, change the sampling rate
  if (fs != 1023e3 and fs > 1023e3):
    ratio = 1023e3 / fs;
    ndx = np.arange(ratio, 1023+ratio, ratio);
    ndx = np.uint16(np.ceil(ndx)) - 1;
    code = code[ndx,:];
  else:
    # Error
    code = [];
  
  # The calling code expects this to be transposed (should probably clean the code up)
  return np.transpose(code);
