from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


#=================Make a nc file to initialize Ku===============
# Shape of input: 53088

dt = xr.Dataset({'foo': (('time','lat', 'lon'), np.random.rand(1, 53088, 1))}, 
            coords={'lat': np.arange(53088),
                    'lon': [1],
                    'time': [1900]})

dt.to_netcdf('tm.nc')