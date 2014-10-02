# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 22:53:41 2014

@author: dave
"""
import sys

# update_progress() : Displays or updates a console progress bar
# Accepts a float between 0 and 1. Any int will be converted to a float.
# A value under 0 represents a 'halt'.
# A value at 1 or bigger represents 100%
def ProgressBar(progress):

  barLength = 10 # Modify this to change the length of the progress bar
  status = ""
  if isinstance(progress, int):
      progress = float(progress)
  if not isinstance(progress, float):
      progress = 0
      status = "error: progress var must be float\r\n"
  if progress < 0:
      progress = 0
      status = "Halt...\r\n"
  if progress >= 1:
      progress = 1
      status = "Done...\r\n"
  block = int(round(barLength*progress));
  text = "\r[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,1), status);
  sys.stdout.write(text);
  sys.stdout.flush();
