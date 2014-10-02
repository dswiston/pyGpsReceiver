# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:25:59 2014

@author: dave
"""

import numpy as np
import threading
import queue
import struct
from support import CalculatePseudoRange
from support import CalcSatPos
from support import CaCodes
from solution import LeastSquareSatPos
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time, sleep
from scipy import signal
from copy import deepcopy
from io import BytesIO
from PIL import Image
from urllib import request
from ProgressBar import ProgressBar
from KalmanFilter import KalmanFilter
from MercatorProjection import MercatorProjection
from IPython.terminal.embed import InteractiveShellEmbed
from rtlsdr import RtlSdr

tstart_stack = [];
def tic():
    tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time() - tstart_stack.pop()))

def SAMPLE_RATE_KHZ(): return 2048

ipshell = InteractiveShellEmbed(banner1='DEBUG', exit_msg='DONE!');

class sat:
    def __init__(self):
        self.trackTimes = np.array([],dtype=np.uint64);
        self.words = np.array([],dtype=np.int8);
        self.measOffset = np.array([0],dtype=np.uint64);
        self.bitOffset = np.array([0],dtype=np.uint64);
        self.subframeNdx = np.array([],dtype=np.uint64);
        self.satFrame = satFrame();
        self.subframeOffset  = np.array([]);
        self.freq = np.array([0]);
        self.freqCor = np.array([0]);
        self.phsOffset = np.array([0]);
        self.code = np.zeros(SAMPLE_RATE_KHZ());
        self.lastBit = np.zeros(1,dtype=np.complex64);
        self.bits = np.array([],dtype=np.complex64);
        self.lastPhsMod = np.zeros(1,dtype=np.complex64);
        self.blkNdx = np.array([0]);
        self.slippedBits = np.array([0]);
        
        
class satFrame:
  def __init__(self):
      self.tlm = np.array([0]);
      self.tow = np.array([0]);
      self.weekNum = np.array([0]);
      self.satAcc = np.array([0]);
      self.satHealth = np.array([0]);
      self.Tgd = np.array([0]);
      self.IODC = np.array([0]);
      self.Toc = np.array([0]);
      self.af2 = np.array([0]);
      self.af1 = np.array([0]);
      self.af0 = np.array([0]);
      self.IODE = np.array([0]);
      self.Crs = np.array([0]);
      self.deltaN = np.array([0]);
      self.M0 = np.array([0]);
      self.Cuc = np.array([0]);
      self.e = np.array([0]);
      self.Cus = np.array([0]);
      self.sqrtA = np.array([0]);
      self.Toe = np.array([0]);
      self.Cic = np.array([0]);
      self.omegaBig = np.array([0]);
      self.Cis = np.array([0]);
      self.i0 = np.array([0]);
      self.Crc = np.array([0]);
      self.omegaSmall = np.array([0]);
      self.omegaDot = np.array([0]);
      self.IDOT = np.array([0]);
  
  
        

def AcquireSatellites(data,codes,fs):
  # Assume only one  period worth of data is being passed in
  numCodes = np.size(codes,0);
  blkSize = np.size(data,0);
  
  # Create a frequency vector
  numFreqs = 500;
  freq = np.asarray([np.linspace(-15e3,15e3,numFreqs)],dtype=np.float32);
  # Create a time vector representing the times associated with each sample
  timeVect = np.asarray([np.linspace(0,blkSize-1,blkSize)],dtype=np.float32);
  timeVect = timeVect / fs;
  # Use the frequency and time vector to create a set of IQ modulation vectors used to modulate the
  # data collected.
  iqMod = np.exp(-1j*2*np.pi*np.dot(freq.T,timeVect),dtype=np.complex64);

  # Perform the "search" for the satellites.  This search includes searching over frequency due to
  # Doppler, delay due to range, and codes due to unique satellite BPSK encoding.
  # To save processing time, the convolution needed for the search will be done using the 
  # "Convolution Theorem" via Fourier transforms of the data.
  # The b-vector/matrix represents the Fourier transformed code search space
  b = np.fft.fft(codes,axis=1);
  # Replicate code search for each frequency to be searched
  b = np.tile(b,(numFreqs,1,1));
  # Modulate the data by each of the frequency modulation vectors.  Need to replicate the data so 
  # that can be done in one nice big efficient operation
  freqModSig = np.tile(data,(numFreqs,1)) * iqMod;
  # The a-vector/matrix represents the Fourier transformed frequency search space
  a = np.fft.fft(freqModSig,axis=1);
  a = np.tile(a,(numCodes,1,1));
  a = np.swapaxes(a,0,1);
  # Perform the convolution (time search) via multiplication of the Fourier transforms.  Don't 
  # forget to take the inverse Fourier transform.
  actMap = np.fft.ifft(a*np.conj(b),axis=2);

  # Perform some crummy statistical analysis to determine which satellites are visible
  maxVals = np.max(np.max(abs(actMap),axis=2),axis=0);
  medianVal = np.median(maxVals);
  activeCodes = np.nonzero(maxVals > medianVal * 1.25);

  numActSats = np.size(activeCodes[0]);
  activeSats = [ sat() for i in range(numActSats)];

  # Loop over each believed active satellite and extract the delay and Doppler associated with it.
  # This will be needed for the demodulation step next.
  for ndx in range(0,numActSats):
    
    activeNdx = activeCodes[0][ndx];
    tmp = np.abs(actMap[:,activeNdx,:]);
    row,col = np.unravel_index(tmp.argmax(), tmp.shape);
    
    activeSats[ndx].freq = freq[0,row];
    activeSats[ndx].blkNdx = col;
    activeSats[ndx].code = codes[activeCodes[0][ndx],:];
    
  print('\n|-----------|----------|---------|');
  print('| Code      | Doppler  | Timing  |');
  print('|-----------|----------|---------|');
  for ndx in range(0,np.size(activeCodes)):
    print('| %i \t | %i \t | %i  \t|' % (activeCodes[0][ndx], round(activeSats[ndx].freq), activeSats[ndx].blkNdx));
  print('|-----------|----------|---------|\n');
  

  return (activeSats)




def DecodeDataStream(activeSats,dataQueue):

  # Number of data slices required to guarantee a full 1-2-3 frame
  # For the initial acquisition phase, must get >36sec worth of data
  # Acq. time/number of blocks
  # Read out 36 seconds worth of contiguous data to acquire the satellite frame.  This length of
  # time guarantees all five subframes have been recorded.
  acqTime = np.array(36000.0);
  acqBlks = np.uint16(np.ceil(acqTime / 128));
  # Create the data vector to store the queue's data in
  data = np.empty(0,dtype=np.complex64);
  # This index is used to record which satellites could not be decoded and need to be deleted
  delNdx = np.array([],dtype=np.uint8);
  # The dataNdx variable keeps the decoding operation coherent between block calls so that the
  # waveform can be properly demodulated.
  dataNdx = np.array([0],dtype=np.uint64);

  for ndx in range(0,acqBlks):
    
    # The BitDecode operation below always leaves a leftover block (2046pts) since the decoding 
    # to handle the case where the code search straddles a block moving forward. The code backs off
    # an additional data point in case the code search straddles a block moving backwards.  During 
    # the first iteration through this loop no leftover block is included (it can't, there is no
    #  older data).
    data = np.concatenate((data[-(SAMPLE_RATE_KHZ()+1):],dataQueue.get()));
    dataQueue.task_done();
    
    # As long as this isn't the first time through the loop, account for the extra sample kept from
    # the previous block for cases where the code search straddles a block moving backward
    # Instead of backing off a block, back-off a block+1.  The code below compensates for the '+1'
    if ndx > 0:
      # Since the buffer will be backed off by one data point above, the blkNdx variables need to be 
      # compensated for the step backward.
      for ndx2 in range(0,np.size(activeSats)):
        activeSats[ndx2].blkNdx += 1;
      # The dataNdx variable also needs to be compensated for the step backward to keep the demod 
      # coherent
      dataNdx -= 1;
    
    # The BitDecode function searches for each active satellite's PRN code in the raw data, attempts
    # to demodulate the BPSK signal, store the bits, and record timing/ranging information.
    (dataNdx,activeSats) = BitDecode(data,activeSats,dataNdx,fs,1);
    
    # Display progress of this ~36sec operation
    ProgressBar(float(ndx)/np.float(acqBlks));
    
  for ndx in range(0,np.size(activeSats)):
    # Convert the bits to words
    activeSats[ndx].words,activeSats[ndx].bitOffset = BitsToWords(activeSats[ndx].bits,ndx);

    # Error checking
    if np.size(activeSats[ndx].words) == 0:
      print('Error in bitstream, deleting from list');      
      delNdx = np.append(delNdx, ndx);
      continue;

    # Search for a preamble
    activeSats[ndx].subframeNdx, activeSats[ndx].words = FindPreamble(activeSats[ndx].words);
    
    # Error checking
    if np.size(activeSats[ndx].subframeNdx) == 0:
      print('Error in bitstream, deleting from list');      
      delNdx = np.append(delNdx, ndx);
      continue;

    # Convert the words from -1/1 to 0/1
    activeSats[ndx].words = ConvertToBinary(activeSats[ndx].words);
    
    # Extract an entire frame (5 subframes) from the data.
    activeSats[ndx].satFrame = ExtractFrames(activeSats[ndx].words,activeSats[ndx].subframeNdx);
    
    # Error checking
    if activeSats[ndx].satFrame.IODE == -1:
      print('Error in bitstream, deleting from list');      
      delNdx = np.append(delNdx, ndx);
      continue;

  if np.size(delNdx) > 0:
    # Delete any satellites that it looked like had an error demodulating
    for delItr in reversed(delNdx):
      del activeSats[delItr];
  
  numSats = len(activeSats);
  
  if numSats == 0:
    return(activeSats);

  # Calculate the offset for each frame in the bit stream.  The bitOffset is the offset from the 
  # first bit decoded to the beginning of the first subframe decoded.  This allows the tracktimes 
  # of each bit to be related to first subframe's TOW and corresponding satellite's position.
  for ndx in range(0,numSats):
    # subframeNdx stores the word (20bits) where the first subframe begins in the data stream
    activeSats[ndx].bitOffset = activeSats[ndx].bitOffset + activeSats[ndx].subframeNdx*20 + 1;   
    
  # For each satellite, store the time aligned (aligned to satellite transmit time, not receiver time) vectors
  (activeSats,minSize) = AlignTrackTimes(activeSats,numSats);
    
  # Don't bother keeping all the track times from the initial acquisition, only keep Xms worth of measurements
  offset = minSize - 125;  # Keep 125ms
  for ndx in range(0,numSats):
    activeSats[ndx].trackTimes = activeSats[ndx].trackTimes[offset:];
    activeSats[ndx].measOffset = offset;
    
  # Convert the overall bit offsets to a relative offset and a block offset amount
  # This makes the tracking code simpler later on
  minSize = activeSats[0].bitOffset;
  for ndx in range(1,numSats):
    minSize = np.minimum(minSize,activeSats[ndx].bitOffset);

  for ndx in range(0,numSats):
    activeSats[ndx].bitOffset = activeSats[ndx].bitOffset - minSize;
    
  return(activeSats);
  

    
    
    
    
    
    
def AlignTrackTimes(activeSats,numSats):

  minSize = np.inf;
  for ndx in range(0,numSats):
    # For each satellite, store the time aligned (aligned to satellite transmit time, not receiver time) vectors
    activeSats[ndx].trackTimes = activeSats[ndx].trackTimes[activeSats[ndx].bitOffset:-1];
    minSize = np.minimum(minSize,np.size(activeSats[ndx].trackTimes));
    
  return (activeSats,minSize);






# Performs the BPSK demodulation.
# Technically this is a Differential Binary Phase Shift Keying (DBPSK) demodulation.
# This doubles the BER at the benefit of being much simpler to implement, no need to 
# lock to the carrier.
def BitDecode(data,activeSats,dataNdx,fs,searchSpace):

  # Decode all the satellites at once
  # Define some up-front items
  numPts = np.size([activeSats[0].code],1);
  numSats = len(activeSats);
  numBits = int(len(data)/numPts)-1;
  sizeData = np.size(data);
  searchVect = np.array(np.zeros(3,dtype=np.complex64));

  # Grab the time vector that corresponds to the location in the buffer
  timeVect = np.linspace(dataNdx,dataNdx+np.size(data),np.size(data))/fs;

  for ndx in range(0,numSats):
    
    # Define and/or reset all the variables used for each satellite
    # Must keep a local copy of searchSpace for each satellite so we can refresh it for each satellite loop
    satSearchSpace = searchSpace; 
    numBit = int(0);
    phsDiff = np.array(np.zeros(numBits,dtype=np.complex64));
    freqEst = np.array(np.zeros(numBits));
    bits = np.array(np.zeros(numBits),dtype=np.complex64);
    trackTimes = np.array(np.zeros(numBits),dtype=np.uint64);
    blkNdx = activeSats[ndx].blkNdx;
    sizeTrackTimes = np.size(activeSats[ndx].trackTimes);
    
    # Create the mixing vector to try and bring the signal to DC (accounts for Doppler).
    # Note that when we update the frequency each block, we must account for discrete phase offsets
    # that result from changing the frequency, otherwise this can drive the demod nuts.
    # Discrete phase offset calculation
    activeSats[ndx].phsOffset = activeSats[ndx].phsOffset + activeSats[ndx].freqCor*timeVect[0]*2*np.pi;
    # Create the mixing vector
    iqModTmp = np.exp(-1j * ( 2*np.pi*activeSats[ndx].freq*timeVect - activeSats[ndx].phsOffset ), dtype=np.complex64);
    # Mix the data to (hopefully) DC
    data2 = data * iqModTmp;
    while(blkNdx+numPts < sizeData):
            
      # Search for the code timing offset.  Try on each side of the predicted time.
      (searchVect, bitNdx) = CodeSearch(data2,blkNdx,numPts,activeSats[ndx].code,satSearchSpace);

      # This is a special case where we are moving across a block border
      if numBit+1 > numBits:
        #print('Slipped bit +1');
        bits = np.append(bits,0);
        trackTimes = np.append(trackTimes,0);
        phsDiff = np.append(phsDiff,0);
        freqEst = np.append(freqEst,0);
        
      # Save the bit and the time offset of the bit        
      bits[numBit] = searchVect[bitNdx];
      # If this if the first time through, uses the offset passed in as the reference time, 
      # otherwise use the previous time as the reference time.
      if numBit == 0 and sizeTrackTimes == 0:
        trackTimes[numBit] = blkNdx + bitNdx - satSearchSpace;
        # Increment the block index
        blkNdx = blkNdx + numPts + bitNdx - satSearchSpace;
        satSearchSpace = 1;
      elif numBit == 0:
        trackTimes[numBit] = activeSats[ndx].trackTimes[-1] + numPts + bitNdx - satSearchSpace;
        # Increment the block index
        blkNdx = blkNdx + numPts + bitNdx - satSearchSpace;
        satSearchSpace = 1;
      else:
        trackTimes[numBit] = trackTimes[numBit-1] + numPts + bitNdx - satSearchSpace;
        # Increment the block index
        blkNdx = blkNdx + numPts + bitNdx - satSearchSpace;
      
      # Increment the bit counter        
      numBit = numBit + 1;
    
    # This is a special case where we are moving across a block border
    if numBit < numBits:
      #print('Slipped bit -1');
      delNdx = np.arange(numBit,np.size(bits));
      bits = np.delete(bits,delNdx);
      trackTimes = np.delete(trackTimes,delNdx);
      phsDiff = np.delete(phsDiff,delNdx);
      freqEst = np.delete(freqEst,delNdx);
      
    # Update the slipped bit running counter
    activeSats[ndx].slippedBits = activeSats[ndx].slippedBits + numBit - numBits;
    
    # Once there is more than one measurement, begin making phase difference and frequency
    # measurements
    phsDiff[1:] = bits[1:] * np.conj(bits[0:-1]);
    if dataNdx > 0:
      phsDiff[0] = bits[0] * np.conj(activeSats[ndx].lastBit);
    else:
      phsDiff[0] = 0;      
    freqEst = np.angle(phsDiff**2);
    
    # Calculate the frequency correction necessary for this satellite
    activeSats[ndx].freqCor = np.mean(freqEst/2) / (2*np.pi) * 1e3;
    activeSats[ndx].freq = activeSats[ndx].freq + activeSats[ndx].freqCor;
    activeSats[ndx].lastBit = bits[-1];
    
    # Update the data block index for "next time"
    activeSats[ndx].blkNdx = blkNdx + numPts - np.size(data);
    
    # Calculate the instantaneous frequency of the BPSK modulated signal
    bpskDemod = np.angle( phsDiff**2 ) / 2;
    # Create a phase modulation vector to remove the frequency (shift signal to DC)
    phsMod = activeSats[ndx].lastPhsMod + np.cumsum(bpskDemod);
    activeSats[ndx].lastPhsMod = phsMod[-1];
    bits = np.angle( bits * np.exp(-1j*phsMod) );
    # Perform scaling to remove DC - should probably figure out how to remove this later
    #bits = bits - ( np.max(bits) + np.min(bits) ) / 2;  
    activeSats[ndx].trackTimes = np.append(activeSats[ndx].trackTimes,trackTimes);
    activeSats[ndx].bits = np.append(activeSats[ndx].bits,bits);

  # Pass back the overall data (time) index in the stream
  dataNdx = dataNdx + numBits * numPts;
  
  return (dataNdx,activeSats);
  
  
  
  
  
  


def BitsToWords(bits,inNdx):

  val = np.array(np.zeros(20),dtype=np.uint64);
  bits = bits - ( np.max(bits) + np.min(bits) ) / 2;

  # Each bit is repeated 20 times.  Perform a matched filter operation.
  # Create the simple filter
  filt = np.ones(20);
  # Perform the filtering
  filtBits = signal.lfilter(filt,1,bits);
  
  # Determine the "phasing" of the words in the bit stream
  for ndx in range(0,20):
    val[ndx] = np.mean(np.abs(filtBits[ndx::20]));
  phsNdx = np.argmax(val);

  # Using the phase index found above to achieve "phase lock", decimate the data and look for bad
  # bits.  If the number of bad bits exceeds the threshold, flag it as a bad decode operation.  The
  # satellite and its data will be removed from processing later.
  # Allow X bits of error per word
  thresh = 17.5*np.pi/2;
  decBits = filtBits[phsNdx::20];
  decBits = decBits[1::];
  badBits = np.abs(decBits) < thresh;
  
  if np.any(badBits):
    print('Warning, bit error detected');
    ipshell();
    decBits = np.array([]);
    return (decBits, phsNdx);

  # Simple conversion to +/-1
  decBits[decBits < 0] = -1;
  decBits[decBits > 0] = 1;
  decBits = np.int8(decBits);
  
  return (decBits, phsNdx);
  
  

def FindPreamble(words):

  # Create the preamble
  preamble = np.array([1,-1,-1,-1,1,-1,1,1],dtype=np.int8);
  
  # Filter the input to find the positions where the preamble is found
  filtWords = np.convolve(words, np.flipud(preamble));
  
  # Reshape, fold over by the size of a subframe.  This should result in one
  # column that contains all perfect correlation.
  tmp = np.reshape( filtWords[0:-np.mod(np.size(filtWords),300)], (-1,300));

  # Decompose the rows and look for the resulting entry that contains the preamble
  tmp2 = np.sum(tmp,0);
  preambleNdx = np.where(np.abs(tmp2) == np.size(tmp,0)*8);
  preambleNdx = preambleNdx[0] - 7;

  # Error check
  if (np.size(preambleNdx)==0):
    print('Could not find preamble!');
    return (preambleNdx, words);
  
  # Error condition - This happens from time to time, not sure why. . .
  if (np.size(preambleNdx)>1):
    preambleNdx = preambleNdx[0];
    
  # Locked onto opposite polarity
  if (tmp2[preambleNdx+7] < 0):
    words = -words;
  
  return (preambleNdx, words); 



def ConvertToBinary(words):
  # Convert from -1/1 to 0/1
  words = (words + 1) / 2;
  return words;


# Extract frame data from the bits.  This big chunk of code follows the GPS data frames specified
# by GLOBAL POSITIONING SYSTEM STANDARD POSITIONING SERVICE SIGNAL SPECIFICATION
# http://www.navcen.uscg.gov/pubs/gps/sigspec/gpssps1.pdf
def ExtractFrames(frameData,subframeNdx):

  # This is the defined pi value that GPS calculations use
  piVal = 3.1415926535898;
  
  # Trim off incomplete frame at the beginning
  frameData = frameData[subframeNdx:-1];
  
  # Trim off incomplete frame at the end
  remainder = np.mod(np.size(frameData),300);
  frameData = frameData[0:-remainder];
  
  # Perform parity checking on the subframes
  frameData = CheckParity(frameData);
  
  # Reshape into a matrix of subframes
  frameData = np.reshape(frameData,(-1,300));
  
  # Record which subframe is which in the vector of subframes 
  subframeNdx = ArrayBinaryToDecimal(frameData[:,[49,50,51]]);
  
  # Find the subframe order, use this to index into the subframe data
  frameNdx = np.argsort(subframeNdx[0:5]);
   
  frame = satFrame();
  frame.tlm = ArrayBinaryToDecimal(frameData[:,0:8]);
  # TOW stores TOW of next subframe, subtract 6 to get the TOW for the beginning of the first
  # subframe collected (not necessarily the first subframe!)
  frame.tow = ArrayBinaryToDecimal(frameData[:,30:47]) * 6 - 6;
  frame.weekNum = ArrayBinaryToDecimal(frameData[frameNdx[0],60:70]);
  frame.satAcc = frameData[frameNdx[0],70:74];
  frame.satHealth = frameData[frameNdx[0],74:80];
  
  # 2.4.3.5 Estimated Group Delay Differential
  # Bits 17 through 24 of word seven contain the correction term, TGD, to account for the effect of
  # satellite group delay differential. Application of the TGD correction term is identified in Section
  # 2.5.5.1.
  frame.Tgd = ArrayBinaryToDecimal(frameData[frameNdx[0],30*6+np.arange(16,24)],-31,0,1);
  
  # 2.4.3.4 Issue of Data, Clock
  # Bits 23 and 24 of word three in subframe 1 are the two MSBs of the ten-bit Issue of Data, Clock
  # (IODC) term; bits one through eight of word eight in subframe 1 will contain the eight LSBs of the
  # IODC. The IODC indicates the issue number of the data set and thereby provides the user with a
  # convenient means of detecting any change in the correction parameters. The transmitted IODC
  # will be different from any value transmitted by the satellite during the preceding seven days. The
  # relationship between the IODC and the IODE (Issue Of Data, Ephemeris) terms are defined in
  # Section 2.4.4.2.
  frame.IODC = ArrayBinaryToDecimal( np.append(frameData[frameNdx[0],30*2+np.arange(22,24)], frameData[frameNdx[0],30*7+np.arange(0,8)]) );
  iode1 = ArrayBinaryToDecimal( frameData[frameNdx[0],30*7+np.arange(0,8)] );
  
  # 2.4.3.6 Satellite Clock Correction Parameters
  # Bits nine through 24 of word eight, bits one through 24 of word nine, and bits one through 22 of
  # word ten contain the parameters needed by the users for apparent satellite clock correction (toc,
  # af2, af1, af0). Application of the clock correction parameters is identified in Section 2.5.5.2.
  frame.Toc = ArrayBinaryToDecimal(frameData[frameNdx[0],30*7+np.arange(8,24)],4);
  frame.af2 = ArrayBinaryToDecimal(frameData[frameNdx[0],30*8+np.arange(0,8)],-55,0,1);
  frame.af1 = ArrayBinaryToDecimal(frameData[frameNdx[0],30*8+np.arange(8,24)],-43,0,1);
  frame.af0 = ArrayBinaryToDecimal(frameData[frameNdx[0],30*9+np.arange(0,22)],-31,0,1);
  
  # 2.4.4.2 Issue of Data, Ephemeris
  # The Issue of Data, Ephemeris (IODE) is an 8 bit number equal to the 8 LSBs of the 10 bit IODC of
  # the same data set. The issue of ephemeris data (IODE) term will provide the user with a
  # convenient means for detecting any change in the ephemeris representation parameters. The
  # IODE is provided in both subframes 2 and 3 for the purpose of comparison with the 8 LSBs of the
  # IODC term in subframe 1. Whenever these three terms do not match, a data set cutover has
  # occurred and new data must be collected. The transmitted IODE will be different from any value
  # transmitted by the satellite during the preceding six hours.
  frame.IODE = ArrayBinaryToDecimal(frameData[frameNdx[1],30*2+np.arange(0,8)]);
  iode2 = ArrayBinaryToDecimal(frameData[frameNdx[1],30*2+np.arange(0,8)]);
  iode3 = ArrayBinaryToDecimal(frameData[frameNdx[2],30*9+np.arange(0,8)]);
  
  if (iode1 != iode2) or (iode2 != iode3) or (iode1 != iode3):
    # An ephemeris/clock update took place between subframes, can't use this data
    print("An ephemeris or clock update took place between subframes, can't use this data!");
    frame.IODE = -1;
    return (frame);
  
  frame.Crs = ArrayBinaryToDecimal(frameData[frameNdx[1],30*2+np.arange(8,24)],-5,0,1);
  frame.deltaN = ArrayBinaryToDecimal(frameData[frameNdx[1],30*3+np.arange(0,16)],-43,0,1) * piVal;
  frame.M0 = ArrayBinaryToDecimal(np.append( frameData[frameNdx[1],30*3+np.arange(16,24)], frameData[frameNdx[1],30*4+np.arange(0,24) ]),-31,0,1) * piVal;
  frame.Cuc = ArrayBinaryToDecimal(frameData[frameNdx[1],30*5+np.arange(0,16)],-29,0,1);
  frame.e = ArrayBinaryToDecimal(np.append( frameData[frameNdx[1],30*5+np.arange(16,24)], frameData[frameNdx[1],30*6+np.arange(0,24)] ),-33);
  frame.Cus = ArrayBinaryToDecimal(frameData[frameNdx[1],30*7+np.arange(0,16)],-29,0,1);
  frame.sqrtA = ArrayBinaryToDecimal(np.append( frameData[frameNdx[1],30*7+np.arange(16,24)], frameData[frameNdx[1],30*8+np.arange(0,24)] ),-19);
  frame.Toe = ArrayBinaryToDecimal(frameData[frameNdx[1],30*9+np.arange(0,16)],4);
  
  frame.Cic = ArrayBinaryToDecimal(frameData[frameNdx[2],30*2+np.arange(0,16)],-29,0,1);
  frame.omegaBig = ArrayBinaryToDecimal(np.append( frameData[frameNdx[2],30*2+np.arange(16,24)], frameData[frameNdx[2],30*3+np.arange(0,24)] ),-31,0,1) * piVal;
  frame.Cis = ArrayBinaryToDecimal(frameData[frameNdx[2],30*4+np.arange(0,16)],-29,0,1);
  frame.i0 = ArrayBinaryToDecimal(np.append( frameData[frameNdx[2],30*4+np.arange(16,24)], frameData[frameNdx[2],30*5+np.arange(0,24)] ),-31,0,1) * piVal;
  frame.Crc = ArrayBinaryToDecimal(frameData[frameNdx[2],30*6+np.arange(0,16)],-5,0,1);
  frame.omegaSmall = ArrayBinaryToDecimal(np.append( frameData[frameNdx[2],30*6+np.arange(16,24)], frameData[frameNdx[2],30*7+np.arange(0,24)] ),-31,0,1) * piVal;
  frame.omegaDot = ArrayBinaryToDecimal(frameData[frameNdx[2],30*8+np.arange(0,24)],-43,0,1) * piVal;
  frame.IDOT = ArrayBinaryToDecimal(frameData[frameNdx[2],30*9+np.arange(8,22)],-43,0,1) * piVal;
  
  return (frame);


# Helper function to convert an array of binary values to a decimal value.  Accepts all sorts of
# flags/formats including power offsets, different endianness, and twos complement formats
def ArrayBinaryToDecimal(inputArray, powOffset = 0, endianness = 0, twosComplement = 0):
  
  if (np.ndim(inputArray) == 1):
    inputArray = np.expand_dims(inputArray,0);
    
  inputLen = np.size(inputArray,1);
    
  if (twosComplement):
    inputLen = inputLen - 1;
  
  if (endianness == 0):
    if (twosComplement):
      offset = -(2**(inputLen+powOffset)) * inputArray[:,0];
      inputArray = inputArray[:,1:];
    else:
      offset = 0;

    powArray = 2**np.arange(powOffset+inputLen-1,powOffset-1,-1,dtype=np.float64);

  else:
    if (twosComplement):
      offset = -(2**inputLen) * inputArray[:,-1];
      inputArray = inputArray[:,1:-1];
    else:
      offset = 0;

    powArray = 2**np.arange(powOffset,inputLen,dtype=np.float64);

  outDec = np.transpose(np.dot(powArray,inputArray.T)) + offset;

  return outDec;



def CheckParity(words):

  for ndx in range(30,np.size(words)-30,30):
    if (np.mod(ndx,300)==0):
      continue;
    else:
      # If bit #30 is set to a “1”, the data in the following word must be inverted
      # The actual parity operation to check integrity is skipped here
      if words[ndx-1]:
        words[np.arange(ndx,ndx+24)] = np.logical_not(words[np.arange(ndx,ndx+24)]);
  
  return words;


# Convert earth-center-earth-fixed (ECEF) cartesian coordinates to spherical latitude, longitude,
# and altitude coordinates
def ecef2lla(x,y,z):

  # Define the WGS-84 ellipsoid constants
  # Earth radius
  a = 6378137
  # Eccentricity
  e = 8.1819190842622e-2
  
  # Begin the dense code-block of calculations
  b = np.sqrt(np.power(a,2) * (1-np.power(e,2)));
  ep = np.sqrt((np.power(a,2)-np.power(b,2))/np.power(b,2));
  p = np.sqrt(np.power(x,2)+np.power(y,2));
  th = np.arctan2(a*z, b*p);
  lon = np.arctan2(y, x);
  lat = np.arctan2((z+ep*ep*b*np.power(np.sin(th),3)), (p-e*e*a*np.power(np.cos(th),3)));
  n = a/np.sqrt(1-e*e*np.power(np.sin(lat),2));
  alt = p/np.cos(lat)-n;
  
  # Convert to degrees
  lat = (lat*180)/np.pi;
  lon = (lon*180)/np.pi;
  
  return (lat, lon, alt);


 

def ConvertData(tmp,blkSize):
  # Unpack the bytes of the string in data into unsigned characters
  readFormat = str(blkSize) + 'B'
  tmp = struct.unpack(readFormat,tmp);
  # Convert to a numpy array of floats
  tmp = np.asarray(tmp,dtype=np.float32);
  # Subtract 127 from the data (to convert to signed)
  tmp = tmp - 127;
  data = np.zeros(len(tmp)/2, dtype=np.complex64);
  data.real = tmp[::2];
  data.imag = tmp[1::2];
  return data
  
class FileReader(threading.Thread):
  def run(self):

    #fid = open('/home/dave/gps_29','rb');  # 2 - OKish, 4 - OK, 5 - OK,   6 - OKish
    fid = open('/home/dave/140320_1243_1.dat','rb');  # 2 - OKish, 4 - OK, 5 - OK,   6 - OKish

    blockLoc = np.zeros(1,dtype=np.uint64);
    # Read 500ms of data at a time
    blkSize = SAMPLE_RATE_KHZ()*256;
    # Acq. time/number of blocks
    # Read out 36 seconds worth of contiguous data to acquire the satellite frame.  This length of
    # time guarantees all five subframes have been recorded.
    acqTime = np.array(36000.0);
    acqBlks = np.uint16(np.ceil(acqTime / (blkSize / SAMPLE_RATE_KHZ() / 2)) + 1); 
    # Add one to account for the initial search routine that uses and then disregards the first
    # block of data
    
    for ndx in range(0,acqBlks):
      data = fid.read(blkSize);
      data = ConvertData(data,len(data));
      dataQueue.put(data);
      sleep(0.256);
    # Only the last 125ms of data from the previous loop is kept from the initial acquisition loop
    # by the track loop. blockLoc represents the offset of the data relative to the first 125ms of
    # data passed to the track loop from above.  Hence set the initial blockLoc value to 125ms
    blockLoc = blockLoc + SAMPLE_RATE_KHZ()*256/SAMPLE_RATE_KHZ()/2
    # Set the block size to 125ms.  The track loop operates on 125ms chunks of data
    blkSize = SAMPLE_RATE_KHZ()*256;
    # Now that we have read enough data to acquire the satellite frame, sample data much less frequently    
    while len(data) >= 1:
      data = fid.read(blkSize);
      data = ConvertData(data,len(data));
      # If the track loop is not ready for the data, do not push it onto the queue.  Continue
      # grabbing new data and dumping the old data until the track queue is ready.
      if (dataQueue.unfinished_tasks == 0):
        dataQueue.put(data);
        dataQueue.put(blockLoc);
        
      sleep(0.256);
      
      # Keep track of the offset from the initial track location so that when the track loop is
      # ready to accept new data it knows how far in time it has skipped 
      blockLoc = blockLoc + blkSize/SAMPLE_RATE_KHZ()/2;





def sdrCallback(samples, sdrQueue):
  
  obj = sdrQueue.get();
  sdrQueue.task_done();
  sdr = obj[0];
  blkLoc = obj[1];
  blkSize = obj[2];
  acq = obj[3];
  breakNdx = np.ceil(36000 / blkSize) * blkSize + 1;
  
  if acq:
    dataQueue.put(samples);
    blkLoc = blkLoc + blkSize;
    if blkLoc >= breakNdx:
      acq = 0;
      blkLoc = blkSize;

  else:
    # If the track loop is not ready for the data, do not push it onto the queue.  Continue
    # grabbing new data and dumping the old data until the track queue is ready.
    if (dataQueue.unfinished_tasks == 0):
      dataQueue.put(samples);
      dataQueue.put(blkLoc);
    # Keep track of the offset from the initial track location so that when the track loop is
    # ready to accept new data it knows how far in time it has skipped 
    blkLoc = blkLoc + blkSize;
    
  sdrQueue.put((sdr,blkLoc,blkSize,acq));







class SdrBuffer(threading.Thread):
  def __init__(self,sdrQueue):
    self.sdrQueue = sdrQueue;
    threading.Thread.__init__(self);
    
  def run(self):
    
    # Read 128ms of data at a time (must be a multiple of a power of two)
    blkSize = SAMPLE_RATE_KHZ()*128;
    blockLoc = np.zeros(1,dtype=np.uint64);
    
    sdr = RtlSdr();
    # configure device
    sdr.sample_rate = SAMPLE_RATE_KHZ()*1e3  # Hz
    sdr.center_freq = 1575420000     # Hz
    sdr.gain = 29;
    sdrQueue.put((sdr,blockLoc,blkSize/SAMPLE_RATE_KHZ(),1));
    sdr.read_samples_async(sdrCallback, blkSize, context=sdrQueue);



      
      
      
      
      
      
      
class InitialAcquisition(threading.Thread):
  def __init__(self, dataQueue, acqQueue, codes, fs):
      self.dataQueue = dataQueue;
      self.acqQueue = acqQueue;
      self.codes = codes;
      self.fs = fs;
      threading.Thread.__init__(self);
        
  def run(self):
    
    dataVector = dataQueue.get();
    dataQueue.task_done();
    # Ignore the first slice of data since the radio's PLL might not be settled
    # (have noticed bad data)
    dataVector = dataVector[-SAMPLE_RATE_KHZ()::];
    activeSats = AcquireSatellites(dataVector,codes,fs);
    activeSats = DecodeDataStream(activeSats,dataQueue);
    if len(activeSats) < 4:
      print("Only decoded %i satellites.  Need at least 4.  Exit." % len(activeSats));
    else:
      acqQueue.put(activeSats);







class TrackLoop(threading.Thread):
  def __init__(self,dataQueue,acqQueue,trackQueue,fs):
      self.dataQueue = dataQueue;
      self.acqQueue = acqQueue;
      self.trackQueue = trackQueue;
      self.fs = fs;
      threading.Thread.__init__(self);
        
  def run(self):

    activeSats = acqQueue.get();
    acqQueue.task_done();
    numSats = len(activeSats);
    dataNdx = 0;
    while(1):
            
      # Put the satellite track times and the data index on the track queue
      trackQueue.put(activeSats);
      trackQueue.put(dataNdx);
      trackQueue.join();
      
      # Grab the latest data collected from the receiver
      dataVector = dataQueue.get();
      # Collect the data's block offset
      dataNdx = dataQueue.get();
      
      minOffset = np.inf;
      for ndx in range(0,numSats):
        
        # Compensate for block crossings.  If the entire bit-stream were available, a slipped bit 
        # would result in a vector of bits smaller or larger relative to the others.  When only 
        # operating on pieces, the slipped bit counter must be used to offset a data vector 
        # relative to the other vectors.
        activeSats[ndx].bitOffset = activeSats[ndx].bitOffset - activeSats[ndx].slippedBits;
        # Keep track of bitOffset to catch the case where a satellite's offset might go negative.
        # If it does go negative, 
        minOffset = np.minimum(minOffset,activeSats[ndx].bitOffset);
        # Skip ahead one block in case the search space causes the bit decoding to slip a block.
        activeSats[ndx].blkNdx = activeSats[ndx].blkNdx + SAMPLE_RATE_KHZ();
        # Clear out the old track times
        activeSats[ndx].trackTimes = np.array([],dtype=np.uint64);
        # Reset the slipped bits field.  Assign slippedBits to '1' because the block offset applied
        # above is going to make the bit decode operation think a bit has been slipped
        activeSats[ndx].slippedBits = 1;    
        
      # If a satellites bit offset goes negative, to line it up with others, each offset must be
      # increased.  Otherwise in the align code below the data vectors cannot be lined up.
      if minOffset < 0:
        for ndx in range(0,numSats):
          activeSats[ndx].bitOffset = activeSats[ndx].bitOffset - minOffset;
      
      # Run the decode code in order to get the track times
      (dummy,activeSats) = BitDecode(dataVector,activeSats,0,fs,50);
      
      # For each satellite, time align (aligned to satellite transmit time, not receiver time) the 
      # track time vectors
      (activeSats,minSize) = AlignTrackTimes(activeSats,numSats);
      
      # Signal to the data stream that we are ready for more data to be put on the queue
      dataQueue.task_done();
      dataQueue.task_done();







class PosCalcLoop(threading.Thread):
  def __init__(self,trackQueue,navQueue):
      self.trackQueue = trackQueue;
      self.navQueue = navQueue;
      threading.Thread.__init__(self);
        
  def run(self):
    while(1):
      
      # Get the latest satellite track times and the data's block offset index
      activeSats = trackQueue.get();
      offsetNdx = trackQueue.get();
      
      # Create some default variables
      numSats = len(activeSats);
      delays = np.zeros(numSats);
      clkCorr = np.zeros(numSats);
      satPos = np.zeros((3,numSats));
    
      # Find the length of the smallest track time buffer.  Not all satellites will have the same
      # number measurements due to timing offsets.  Only loop over the timing points present for
      # each satellite.
      minSize = np.size(activeSats[0].trackTimes);
      for ndx in range(1,numSats):
        minSize = np.minimum(minSize,np.size(activeSats[ndx].trackTimes));   
    
      # Step over every other measurement to save time
      # Calculate the results (pos) size, depends on if there is an odd or even number of 
      # measurements
      if np.mod(minSize,2):
        pos = np.zeros((4,minSize/2+1));
      else:
        pos = np.zeros((4,minSize/2));
        
      # Loop over each timing measurement, step by two (every other) to save processing time
      for measNdx in range(0,minSize,2):
        
        # For each satellite, calculate the position of the satellite at the measurement time
        for satNdx in range(0,numSats):
          # Record the position in the timeline of each satellite for identically generated time 
          # coincident bits.
          delays[satNdx] = activeSats[satNdx].trackTimes[ measNdx ] + 2;
          # From the fully decoded frame, calculate the satellite position when the bit was 
          # generated.  Each unique measurement index is 1ms of duration/delay.
          satPos[:,satNdx], clkCorr[satNdx] = CalcSatPos( activeSats[satNdx].satFrame, activeSats[satNdx].satFrame.tow[0]+(activeSats[satNdx].measOffset+offsetNdx+measNdx)/1000 );

        # Using the different delays associated with each satellite for the same bit, calculate the pseudo range
        psuedoRange = CalculatePseudoRange(delays,fs);
                
        # For each set of pseudoRanges, calculate the receiver position
        (pos[:,measNdx/2]) = LeastSquareSatPos(satPos, psuedoRange + clkCorr * 2.99792458e8);

      navQueue.put(pos);
      navQueue.put(offsetNdx);
      navQueue.join();
      trackQueue.task_done();
      trackQueue.task_done();



      



# Code searches a data vector for correlation with a particular gold code.  Uses a particular
# search window size.  Essentially performing pulse compression, searching for the correlation 
# spike, and recording the spike's position (timing) and the value (the data bit) 
def CodeSearch(data,blkNdx,numPts,code,searchSpace):
  
  searchVect = np.array(np.zeros(searchSpace*2+1),dtype=np.complex64);

  # Search for the code timing offset.  Try on each side of the predicted time.
  for ndx in range(-searchSpace,searchSpace+1):
    searchVect[ndx+searchSpace] = np.inner(data[blkNdx+ndx:blkNdx+numPts+ndx],code);
    
  # Determine which code timing value has the highest correlation
  bitNdx = np.argmax(np.abs(searchVect));

  return (searchVect,bitNdx);








class MapLoop(threading.Thread):
  def __init__(self,navQueue,mapQueue):
      self.navQueue = navQueue;
      self.mapQueue = mapQueue;
      threading.Thread.__init__(self);
        
  def run(self):
    
    # Create the Kalman filter tracker
    A = np.eye(6);
    deltaT = np.array(2e-3);
    A2 = np.eye(6,k=3) * deltaT;
    A = A + A2;
    R = np.eye(3) * 100**2;
    H = np.eye(3,6);
    P = np.eye(6) * 100**2;
    B = np.eye(6);
    
    kalman = KalmanFilter(A,R,H,P,B);
    
    first = 0;
    oldDataNdx = np.zeros(1,dtype=np.uint64);
    tmp = np.array(np.zeros(6));

    while(1):
      
      navData = navQueue.get();
      dataNdx = navQueue.get();

      numMeas = np.size(navData,1);
      
      if first == 0:
        kalman.initialize( np.hstack( (navData[0:3,0],0,0,0) ) );
            
      # Loop over each position measurement
      for measNdx in range(0,numMeas):
        if measNdx == 0 and first != 0:
          deltaT = np.array(np.float(dataNdx - oldDataNdx)*1e-3);
          A2 = np.eye(6,k=3) * deltaT;
          kalman.A = A + A2;
        else:
          deltaT = np.array(1e-3);
          A2 = np.eye(6,k=3) * deltaT;
          kalman.A = A + A2;
        
        kalman.run( navData[0:3,measNdx] );
        tmp = np.vstack((tmp,kalman.x));

      print('%i' % (dataNdx - oldDataNdx));
      oldDataNdx = deepcopy(dataNdx);

      navQueue.task_done();
      navQueue.task_done();

      first = first + 1;
      # Convert from ECEF to lat/lon
      (lat, lon, height) = ecef2lla(kalman.x[0],kalman.x[1],kalman.x[2]);
      mapQueue.put((lat,lon));
#      print('%.10f, %.10f, %.10f' % (lat, lon, height));
#      print('%.10f, %.10f, %.10f' % (kalman.x[3], kalman.x[4], kalman.x[5]));
        






      
# This queue stores the raw IQ data from the receiver
dataQueue = queue.Queue();
# This queue stores the acquired satellite buffer, includes all acquired satellite data
acqQueue = queue.Queue();
# This queue stores the track data
trackQueue = queue.Queue();
# This queue stores the nav data to be displayed on a map
navQueue = queue.Queue();
# This queue passes the map image and kalman filter lat/lon data used for plotting
mapQueue = queue.Queue();
sdrQueue = queue.Queue();


fs = SAMPLE_RATE_KHZ()*1e3;
# Generate the 37 gold codes used by each satellite  
#codes = np.genfromtxt('caCodes.csv', delimiter=',', dtype=np.float32);
codes = CaCodes(fs);
codes = np.exp(1j*np.pi*codes);
# Data gathering loop
fileRead = FileReader();
fileRead.start();
# Initial acquisition
initialAcq = InitialAcquisition(dataQueue,acqQueue,codes,fs);
initialAcq.start();
# Position Loop
posCalcLoop = PosCalcLoop(trackQueue,navQueue);
posCalcLoop.start();
# Tracking Loop
trackLoop = TrackLoop(dataQueue,acqQueue,trackQueue,fs);
trackLoop.start();
# Map Display Loop
mapLoop = MapLoop(navQueue,mapQueue);
mapLoop.start();
# Finally, start the RTL-SDR interface thread
#rtlSdr = SdrBuffer(sdrQueue);
#rtlSdr.start();







(centerLat,centerLon) = mapQueue.get();
url = "http://maps.googleapis.com/maps/api/staticmap?center=" + str(centerLat) + "," + str(centerLon) + "&size=800x800&zoom=19&sensor=false"
buffer = BytesIO(request.urlopen(url).read());
imgData = Image.open(buffer);




obj = MercatorProjection();
zoom = 19;
mapWidth = 640;
mapHeight = 640;
corners = obj.getCorners(centerLat, centerLon, zoom, mapWidth, mapHeight);
ims = [];

plt.ion();
plt.rcParams['toolbar'] = 'None';
fig = plt.figure(num=1,frameon=False,figsize=(8, 8));
ax = plt.axes([0,0,1,1]);
plt.axis("off");
#plt.xlim((0,640));
#plt.ylim((640,0));
im = plt.imshow(imgData,interpolation='lanczos');
ims.append([im]);
plt.hold(True);

for ndx in range(0,100):

  (pointLat,pointLon) = mapQueue.get();
  pixelY = (pointLon - centerLon) / (corners[1] - corners[3]) * mapHeight + mapHeight/2;
  pixelX = (pointLat - centerLat) / (corners[0] - corners[2]) * mapWidth + mapWidth/2;
  print((pointLat,pointLon));

  im = plt.plot(pixelY+1,640-pixelX+1,'o');
  plt.draw();
  plt.savefig('/home/dave/gpspics/tmp' + str(ndx) + '.png', transparent=True, bbox_inches='tight',pad_inches=0);

  

    
    
    