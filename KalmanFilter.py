# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 20:14:44 2014

@author: dave
"""

import numpy as np

class KalmanFilter:
  
  x = [];  # Kalman filter predictions.  Updated each filter iteration (step).
           # Size n
  z = [];  # Kalman filter inputs (measurements)
           # Size m
  R = [];  # Measurement noise covariance matrix.  Generally held as a constant and not updated over
           # time (steps).
           # Size m x m
  A = [];  # Difference equation, relates the state at the previous step to the state at the 
           # current step.  This is a constant, the relationship between the current and the next 
           # state does not vary over time (steps).
           # Size n x n
  H = [];  # Measurement equation that relates the state to the measurements.  Since the measurement
           # has a non-linear relationship to the state, this is a Jacobian matrix for the function 
           # relating the measurements to the state.  This is updated/calculated at each step.
           # Size m x n
#  h = [];  # Measurement equation that relates the state to the measurements.  In the filter
#           # implementation this is a non-linear equation.
#           # Size
  P = [];  # Estimate error (states) covariance matrix.  This is calculated/updated at each step.
           # Size n x n
  K = [];  # Kalman filter gain or "blending" factor that minimizes the a-posteriori error 
           # covariance.  This is calculated/updated at each step.
           # Size n x m
  u = [];  # Control signal
           # Size l
  B = [];  # This matrix relates the control input to the state.  Movement of the platform will
           # cause the state to change.
           # Size n x l
  
  likelihood = np.array([]);
  
  def __init__(self,A,R,H,P,B):
    n = np.size(A,0);
    m = np.size(R,0);
    l = np.size(B,1);

    self.A = A;
    self.R = R;
    self.H = H;
    self.P = P;
    self.B = B;

    self.x = np.array(np.zeros(n));
    #self.z = np.array(np.zeros(m));
    #self.h = np.array(np.zeros(0));
    self.K = np.array(np.zeros((n,m)));
    self.u = np.array(np.zeros(l));
    self.likelihood = np.array(np.zeros(n));

        
        
  def run(self,meas):
          
    # If there is a control signal, update it based on the control input
    # u = CalculateControlSignal(control,kalman.x(:,end));
    u = np.zeros(6);
    
    # Perform the prediction updates
    xP = np.dot(self.A, self.x) + np.dot(self.B, u);
    pP = np.dot(self.A, np.dot(self.P, np.transpose(self.A)));
    
    # # Evaluate/calculate the measurement function and its Jacobian matrix at the predicted state
    # if isempty(self.H)
    #   [self.H self.h] = CalcMeasEq(xP,'otherInfoNeeded');
    # else
    #   [self.H(:,:,end+1) self.h(:,end+1)] = CalcMeasEq(xP,'otherInfoNeeded');
    # end
    
    # Perform measurement updates
    # Update the filter gain
    self.K = np.dot(pP, np.dot(np.transpose(self.H), np.linalg.inv( np.dot(self.H, np.dot(pP, np.transpose(self.H))) + self.R )));
    # Make a new state prediction
    # self.x(:,end+1) = xP + self.K(:,:,end) * ( meas - self.h(:,end) );
    self.x = xP + np.dot(self.K, ( meas - np.dot(self.H, xP)));
    
    
    # Update the state covariance matrix
    self.P = np.dot(np.eye(np.size(self.P,0),np.size(self.P,1)) - np.dot(self.K, self.H), pP);
    
    # Kalman filter stability scoring
    innov = meas - np.dot(self.H,xP);
    innovCov = np.dot(self.H, np.dot(pP, np.transpose(self.H))) + self.R;
    self.likelihood = np.exp(-0.5 * np.transpose(innov) * np.dot(np.linalg.inv(innovCov), innov)) / (np.sqrt(np.power(2*np.pi,3) * np.linalg.det(innovCov)));


  def initialize(self,initStates):
    self.x = initStates;