# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 16:56:07 2014

@author: dave
"""

from __future__ import division
import math

class MercatorProjection :

  def __init__(self) :
    self.MERCATOR_RANGE = 256;
    self.originX = self.MERCATOR_RANGE / 2;
    self.originY = self.MERCATOR_RANGE / 2;
    self.lat = 0;
    self.lon = 0;
    self.pixelsPerLonDegree = self.MERCATOR_RANGE / 360;
    self.pixelsPerLonRadian = self.MERCATOR_RANGE / (2 * math.pi);
    

  def  bound(self, value, minVal, maxVal):
    value = max(value, minVal)
    value = min(value, maxVal)
    return value
    

  def  degreesToRadians(self,deg) :
    return deg * (math.pi / 180)
  
  
  def  radiansToDegrees(self,rad) :
    return rad / (math.pi / 180)


  def fromLatLngToPoint(self, lat, lon) :
    
    x = 0;
    y = 0;
    
    x = self.originX + lon * self.pixelsPerLonDegree
    
    # Truncating to 0.9999 effectively limits latitude to
    # 89.189.  This is about a third of a tile past the edge of the world tile.
    siny = self.bound(math.sin(self.degreesToRadians(lat)), -0.9999, 0.9999)
    y = self.originY + 0.5 * math.log((1 + siny) / (1 - siny)) * - self.pixelsPerLonRadian;
    return (x,y);


  def fromPointToLatLng(self,x,y) :
    lon = (x - self.originX) / self.pixelsPerLonDegree;
    latRadians = (y - self.originY) / -self.pixelsPerLonRadian;
    lat = self.radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
    return (lat, lon);


  def getCorners(self, centerLat, centerLon, zoom, mapWidth, mapHeight):
    scale = 2**zoom;
    (x,y) = self.fromLatLngToPoint(centerLat,centerLon);
    swX = x-(mapWidth/2)/scale;
    swY = y+(mapHeight/2)/scale;
    (swLat,swLon) = self.fromPointToLatLng(swX,swY);
    neX = x+(mapWidth/2)/scale;
    neY = y-(mapHeight/2)/scale;
    (neLat,neLon) = self.fromPointToLatLng(neX,neY);
    return (neLat,neLon,swLat,swLon);
