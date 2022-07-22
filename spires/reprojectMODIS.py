'''
test out a reprojection of MODIS HDF
'''

import os
import re
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Transformer
from affine import Affine
from rioxarray.rioxarray import affine_to_coords

#function to plot RGB
def RGBplotRaster(X):
    fig, axs = plt.subplots()
    plt.imshow(X[:,:,[0,3,2]])
    axs.set_title(desc+'\n'+name)
    fig.canvas.draw()

# target EPSG
targetProj="EPSG:3310"
# find modis file in basedir
sep=os.path.sep
basedir="/Users/nbair/Work/code/spyderTest"


d=os.listdir(basedir)
pattern=re.compile(".*.hdf")

for name in d:
    if pattern.search(name):
        fname=basedir+sep+name
        
#open hdf file as xarray dataset              
rds=rxr.open_rasterio(fname,parse_coordinates=False)

#list of layers from 2400x2400 layers
ind=1 #0 is 1200x1200 data, 1 is 2400x2400 data
nBands=7 #needs to be specified a a priori      
lyrs=list(rds[ind].data_vars.keys()) #create a list of dataset keys
crs_str=rds[1].spatial_ref.crs_wkt

#transform lat lon returned to sinusoid
transformer = Transformer.from_crs(targetProj, crs_str, always_xy=True)
west, north = transformer.transform(rds[ind].WESTBOUNDINGCOORDINATE, 
                                    rds[ind].NORTHBOUNDINGCOORDINATE)
#create a transformation    
pixel_size = rds[ind].CHARACTERISTICBINSIZE500M
transform = Affine(pixel_size, 0, west, 0, -pixel_size, north)

coords = affine_to_coords(transform, rds[ind].rio.width, rds[ind].rio.height)
rds[ind].coords["x"] = coords["x"]
rds[ind].coords["y"] = coords["y"]
rds[ind].rio.write_crs(crs_str, inplace=True)

#create m x n x 7 band reprojected dataset
pattern=re.compile("sur.*")
i=0
for lyr in lyrs:
    if pattern.search(lyr):
        data=rds[ind][lyr]
        data=data.squeeze();
        #add null values and scale
        scalefactor=data.attrs["scale_factor"]
        fillvalue=data.attrs["_FillValue"]
        data.values = data.values.astype('float64')
        #set fill values to NaN
        t=(data.values==fillvalue)
        data.values[t]=np.NaN
        #set values < 0 to 0
        t=data.values<0
        data.values[t]=0
        #scale 
        data.values=data.values/scalefactor
        #store original
        data0=data
        # reproject to target
        data = data.rio.reproject(targetProj,nodata=np.nan)
        # allocate new arrays if 1st pass
        if i==0:
            X0=np.zeros([data0.rio.height,data0.rio.width,nBands],'float64')
            X=np.zeros([data.rio.height,data.rio.width,nBands],'float64')
        # fill arrays    
        X0[:,:,i]=data0.values 
        X[:,:,i]=data.values    
        i = i+1
#plot old and new        
desc='unmodified'
RGBplotRaster(X0)

desc=targetProj
RGBplotRaster(X)

