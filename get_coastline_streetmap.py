import numpy as np
import glob
import csv
import ogr
import os
import pickle
#import gdal
#import osr
import matplotlib.pyplot as plt
#import netCDF4 as nc
#from boost.utils.geodesic import distance
#from lxml import etree as ET

#from skimage.draw import polygon
#from skimage.morphology import binary_dilation


# Bretagne
lon_min = -7
lon_max = -4
lat_min = 47
lat_max = 50



shp_filename = '/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/coastlines-split-4326/lines.shp' 
driver = ogr.GetDriverByName('ESRI Shapefile')
coastline_shp = driver.Open(shp_filename)
layer = coastline_shp.GetLayer()

coastline_poly = []
count = 0
for i in range(0,layer.GetFeatureCount()):
    print(count)
    count += 1
    InFeature = layer.GetNextFeature()
    try:
#        coords = InFeature.geometry().GetEnvelope() #(-7.329818, -7.328612, 57.5305504, 57.5321179)
        g = InFeature.geometry()
        if (g.GetGeometryName() == 'LINESTRING'):
            boundary = g.GetPoints()
        elif (g.GetGeometryName() == 'POLYGON'):
            boundary = InFeature.geometry().GetBoundary().GetPoints()
        for p in boundary:
            lon = p[0]
            lat = p[1]
            if lon >= lon_min and lon <= lon_max and lat >= lat_min and lat <= lat_max:
                coastline_poly.append(boundary)
                break  
#        coastline_poly.append(boundary)
    except:
        continue


## Remove all the points outside of ROI
#count = 0
#for x in coastline_poly:
#    print count, len(x)
#    count += 1
#    for p in x:
#        lon = p[0]
#        lat = p[1]
#        if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
#            x.remove(p)

# Plot
for x in coastline_poly:
    poly = np.array(x)
    plt.plot(poly[:,0],poly[:,1])
#plt.xlim([0,15])
#plt.ylim([-6,4])
plt.show()

# Save
filename = '/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/coastlines-split-4326/streetmap_coastline_Bretagne.pkl'
with open(filename, 'w') as f:   #Pickling
    pickle.dump(coastline_poly, f)
    
## Check distance
#dist_max = 0
#tmp = [] 
#for x in coastline_poly:
#    for i in range(len(x) - 1):
#        d = distance(x[i][1], x[i][0], x[i+1][1], x[i+1][0])
#        print d
#        if (d > 4000):
#            tmp.append(x)

#with open(filename + '.txt', 'r') as fp:
#    t = pickle.load(fp)          
            
