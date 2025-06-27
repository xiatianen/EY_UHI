#!/usr/bin/env python
# coding: utf-8

# ## Load In Dependencies

# In[1]:


# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Core Data Science
import numpy as np
import pandas as pd
from scipy import stats,ndimage
from scipy.stats import boxcox,pearsonr,spearmanr
import time
from math import ceil

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Geospatial Analysis
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from osgeo import gdal, osr
from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Point

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

# Utilities
import os
import re
import glob
from tqdm import tqdm
import json
import joblib
import sys
import concurrent.futures


# ## Path settings

# In[2]:


# Get the working directory of the Notebook and dynamically calculate the project root directory (two levels up)
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))

# Add the project root directory to the module search path so Python can dynamically find the config file
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the global instance path_config from path_config.py and rename it as config
from path_config import path_config as config


# ## Dependent Variable

# In[3]:


# Load the training data from csv file and display the first few rows to inspect the data
ground_df = pd.read_csv(config.uhi_path)
ground_df.head()


# ## Rasters resample

# In[4]:


def get_spatial_resolution(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            return dataset.res 
    except Exception as e:
        return f"無法讀取 {file_path}: {e}"

for file in config.file_paths_list:
    resolution = get_spatial_resolution(file)
    print(f"{os.path.basename(file)} 的空間解析度: {resolution}")


# ## Parallel Processing for Multi-Resolution GeoTIFF Focal Mean Calculation and Export

# In[5]:


def export_geotiff_resolutions(
    input_path: str,
    output_folder: str,
    min_res: float,
    max_res: float,
    step: float,
    num_workers: int = 10
):
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the input GeoTIFF
    dataset = gdal.Open(input_path)
    if not dataset:
        raise FileNotFoundError(f"Unable to open the input file: {input_path}")
    
    # Read all necessary data first
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    
    # Obtain spatial reference information and resolution
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    # Assume positive pixel size; use geotransform[1] (horizontal pixel width)
    pixel_size = abs(geotransform[1])
    
    # Close the input file (release resources early)
    dataset = None
    
    # Create the resolution list
    resolutions = np.arange(min_res, max_res + step/2, step)
    
    def process_single_resolution(current_res):
        """Function to process a single resolution for parallel execution"""
        # Construct output filename based on the current radius
        output_filename = f"{config.env_name}_res{int(current_res) if current_res == int(current_res) else current_res}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        # Calculate the radius in pixel units
        radius_pixels = current_res / pixel_size
        # Define the convolution kernel size, using an odd number to ensure center alignment
        kernel_size = int(2 * ceil(radius_pixels) + 1)
        center = ceil(radius_pixels)
        
        # Create a circular convolution kernel (footprint)
        y, x = np.ogrid[-center:kernel_size-center, -center:kernel_size-center]
        mask = x*x + y*y <= radius_pixels*radius_pixels
        footprint = mask.astype(np.float32)
        
        # Compute the focal mean using convolution: handle nan values first
        data_no_nan = np.nan_to_num(arr, nan=0.0)
        sum_arr = ndimage.convolve(data_no_nan, footprint, mode='reflect')
        # Simultaneously compute the count of valid (non-nan) values in each neighborhood
        valid_mask = np.where(np.isnan(arr), 0, 1)
        count_arr = ndimage.convolve(valid_mask, footprint, mode='reflect')
        
        with np.errstate(divide='ignore', invalid='ignore'):
            focal_avg = sum_arr / count_arr
            focal_avg[count_arr == 0] = np.nan
        
        # Write the focal mean result into a new GeoTIFF file
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(
            output_path,
            arr.shape[1],  # Width
            arr.shape[0],  # Height
            1,
            gdal.GDT_Float32,
        )
        out_dataset.SetGeoTransform(geotransform)
        out_dataset.SetProjection(projection)
        out_band = out_dataset.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)
        out_band.WriteArray(focal_avg)
        out_band.FlushCache()
        out_dataset = None
        
        return current_res
    
    # Perform parallel processing for multiple resolution computations
    print(f"Starting to process {len(resolutions)} resolution files using {num_workers} worker threads...")
    
    with tqdm(total=len(resolutions), desc="Focal Mean Processing", unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_res = {executor.submit(process_single_resolution, res): res for res in resolutions}
            
            # Process the completed tasks
            for future in concurrent.futures.as_completed(future_to_res):
                res = future_to_res[future]
                try:
                    result = future.result()
                    pbar.update(1)
                except Exception as exc:
                    print(f"Error processing resolution {res}: {exc}")
    
    print("All processing is complete!")


# In[6]:


# #This will run for a long time, don't drive it.
# for file_path, env_name in zip(config.file_paths_list, config.env_names):
#     export_geotiff_resolutions(file_path, os.path.join(config.resampled_rasters_path, env_name), 30, 1650, 50)
#     print(f"{os.path.basename(file_path)} 的空間解析度: {resolution}")


# ## Find the best resolution

# ### Optimal Resolution Finder via Spearman Correlation Analysis

# In[ ]:


def find_best_resolution_correlation(resampled_path, ground_df, raster_epsg, figures_name):    
    # Store correlations for each resolution
    correlations = {}
    resolution_names = []  # Used to store file names (for X axis)
    correlation_values = []  # Used to store correlation values (for Y axis)
    max_abs_correlation = -1  # Track the current maximum absolute correlation, initialized to -1
    
    # Set up coordinate transformation
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)  # WGS84
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(raster_epsg)  # UTM Zone 18N
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    
    # Get all tiff files and sort by numerical resolution value
    tiff_files = glob.glob(os.path.join(resampled_path, "*.tif"))
    # Extract the numerical part after "res" and sort numerically
    tiff_files.sort(key=lambda x: int(re.search(r'res(\d+)', os.path.basename(x)).group(1)))
    
    # Process all tiff files
    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)
        
        dataset = gdal.Open(tiff_file)
        if dataset is None:
            continue
        
        geo_transform = dataset.GetGeoTransform()
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        values, ground_values = [], []
        
        # Convert coordinates and extract values
        for _, row in ground_df.iterrows():
            x, y, _ = transform.TransformPoint(row['Longitude'], row['Latitude'])
            px = int((x - geo_transform[0]) / geo_transform[1])
            py = int((y - geo_transform[3]) / geo_transform[5])
            
            if 0 <= px < xsize and 0 <= py < ysize:
                try:
                    data = dataset.GetRasterBand(1).ReadAsArray(px, py, 1, 1)
                    if data is not None and not np.isnan(data).any():
                        values.append(float(data[0][0]))
                        ground_values.append(float(row['UHI Index']))
                except Exception:
                    continue
        
        # Calculate Spearman rank correlation if enough valid points are available
        if len(values) > 1:
            try:
                correlation, _ = spearmanr(values, ground_values)
                correlations[filename] = correlation
                resolution_names.append(filename)  # Collect file names
                correlation_values.append(correlation)  # Collect correlation values
                
                # Only print processing information and correlation when the absolute correlation improves
                abs_correlation = abs(correlation)
                if abs_correlation > max_abs_correlation:
                    max_abs_correlation = abs_correlation
                    print(f"\nProcessing: {filename}")
                    print(f"Correlation improved: {correlation:.4f} (Absolute value: {abs_correlation:.4f})")
            except Exception:
                pass
        
        dataset = None
    
    # Find and report the best resolution based on absolute correlation value
    if correlations:
        best_resolution = max(correlations.items(), key=lambda x: abs(x[1]))
        print(f"\nBest correlation found:")
        print(f"Resolution: {best_resolution[0]}")
        print(f"Correlation: {best_resolution[1]:.4f} ({'positive' if best_resolution[1] > 0 else 'negative'})")
        
        # Extract the resolution numbers for the X-axis labels
        x_labels = [int(re.search(r'res(\d+)', name).group(1)) for name in resolution_names]
        
        # Plot the line chart
        plt.figure(figsize=(12, 6))  # Increase the chart width
        plt.plot(x_labels, correlation_values, marker='o', linestyle='-', color='b', label='Correlation')
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # Add a reference line at Y=0
        plt.title('Correlation Across Different Resolutions')
        plt.xlabel('Resolution (meters)')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(x_labels[::2], rotation=45, ha='right')  # Display labels every two ticks, rotated 45 degrees
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the image to figures_path
        plt.savefig(os.path.join(config.figures_path, f'correlation_plot_{figures_name}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return best_resolution
    return (None, None)


# ### Optimal Resolution Finder Using Random Forest Feature Importance

# In[ ]:


def find_best_resolution_correlation(resampled_path, ground_df, raster_epsg, figures_name):    
    # Store feature importance for each resolution (originally used correlation, now using Random Forest importance)
    correlations = {}
    resolution_names = []  # Used to store file names (for the X axis)
    correlation_values = []  # Used to store importance values (for the Y axis)
    max_abs_correlation = -1  # Track the current maximum importance, initialized to -1
    
    # Set up coordinate transformation
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)  # WGS84
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(raster_epsg)  # UTM Zone 18N
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    
    # Get all tiff files and sort by numerical resolution value
    tiff_files = glob.glob(os.path.join(resampled_path, "*.tif"))
    # Extract the numerical part after "res" and sort numerically
    tiff_files.sort(key=lambda x: int(re.search(r'res(\d+)', os.path.basename(x)).group(1)))
    
    # Construct the feature matrix using a dictionary, where each key is a resolution file name and the value is the value corresponding to each record in ground_df (if invalid, remains np.nan)
    features = {}
    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)
        dataset = gdal.Open(tiff_file)
        if dataset is None:
            continue
        geo_transform = dataset.GetGeoTransform()
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        
        # Set np.nan as the default for each record in ground_df
        features[filename] = [np.nan] * len(ground_df)
        
        for i, (_, row) in enumerate(ground_df.iterrows()):
            x, y, _ = transform.TransformPoint(row['Longitude'], row['Latitude'])
            px = int((x - geo_transform[0]) / geo_transform[1])
            py = int((y - geo_transform[3]) / geo_transform[5])
            
            if 0 <= px < xsize and 0 <= py < ysize:
                try:
                    data = dataset.GetRasterBand(1).ReadAsArray(px, py, 1, 1)
                    if data is not None and not np.isnan(data).any():
                        features[filename][i] = float(data[0][0])
                except Exception:
                    continue
        
        dataset = None
    
    # Construct the feature matrix DataFrame (each column represents a resolution)
    X = pd.DataFrame(features)
    # Drop rows with missing values (because Random Forest cannot handle missing values)
    X = X.dropna()
    # Extract the corresponding target variable for the remaining rows
    y = ground_df.loc[X.index, 'UHI Index']
    
    if X.empty:
        return (None, None)
    
    # 使用隨機森林回歸進行多次訓練，重複 100 次並取平均特徵重要性
    runs = 10
    total_importances = np.zeros(len(X.columns))
    for run in range(runs):
        # 使用不同的 random_state 保持結果可重現
        rf = RandomForestRegressor(n_estimators=100, random_state=42 + run)
        rf.fit(X, y)
        total_importances += rf.feature_importances_
    avg_importances = total_importances / runs
    
    # Fill the correlations dictionary and the resolution_names and correlation_values lists
    for col, imp in zip(X.columns, avg_importances):
        correlations[col] = imp
        resolution_names.append(col)
        correlation_values.append(imp)
        if imp > max_abs_correlation:
            max_abs_correlation = imp
            print(f"\nProcessing: {col}")
            print(f"Feature importance improved: {imp:.4f}")
    
    # Find the best resolution (highest feature importance)
    best_resolution = max(correlations.items(), key=lambda x: x[1])
    print(f"\nBest feature importance found:")
    print(f"Resolution: {best_resolution[0]}")
    print(f"Feature Importance: {best_resolution[1]:.4f}")
    
    # Extract resolution numbers for the X-axis labels
    x_labels = [int(re.search(r'res(\d+)', name).group(1)) for name in resolution_names]
    
    # Plot a line chart (maintaining the original chart format)
    plt.figure(figsize=(12, 6))  # Increase the chart width
    plt.plot(x_labels, correlation_values, marker='o', linestyle='-', color='b', label='Correlation')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.title('Correlation Across Different Resolutions')
    plt.xlabel('Resolution (meters)')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(x_labels[::2], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the image to figures_path (this variable is defined beforehand)
    plt.savefig(os.path.join(config.figures_path, f'rf_feature_importance_plot_{figures_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_resolution


# ## Optimal Resolution

# ### Optimal Resolution for Environmental Indicator

# In[ ]:


best_res_b1, max_corr_b1 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b1"), ground_df,32618,"b1")
# 330
# rf780


# In[ ]:


best_res_b2, max_corr_b2 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b2"), ground_df,32618,"b2")
# 380
# rf780


# In[ ]:


best_res_b3, max_corr_b3 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b3"), ground_df,32618,"b3")
# 380
# rf780


# In[ ]:


best_res_b4, max_corr_b4 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b4"), ground_df,32618,"b4")
# 380
# rf780


# In[ ]:


best_res_b5, max_corr_b5 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b5"), ground_df,32618,"b5")
# 330
# rf780


# In[ ]:


best_res_b6, max_corr_b6 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b6"), ground_df,32618,"b6")
# 330
# rf380


# In[ ]:


best_res_b7, max_corr_b7 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b7"), ground_df,32618,"b7")
# 330
# rf480


# In[ ]:


best_res_b8, max_corr_b8 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b8"), ground_df,32618,"b8")
# 330
# rf780


# In[ ]:


best_res_b8a, max_corr_b8a = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b8a"), ground_df,32618,"b8a")
# 330
# rf780


# In[ ]:


best_res_b9, max_corr_b9 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b9"), ground_df,32618,"b9")
# 530
# rf730


# In[ ]:


best_res_b10, max_corr_b10 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b10"), ground_df,32618,"b10")
# 1480
# rf1630


# In[ ]:


best_res_b11, max_corr_b11 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b11"), ground_df,32618,"b11")
# 330
# rf880


# In[ ]:


best_res_b12, max_corr_b12 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"b12"), ground_df,32618,"b12")
# 330
# rf780


# In[ ]:


best_res_dem, max_corr_dem = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"dem"), ground_df,32618,"dem")
# 930
# rf580


# In[ ]:


best_res_forest_hight, max_corr_forest_hight = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"forest_hight"), ground_df,32618,"forest_hight")
# 330
# rf180


# In[ ]:


best_res_greeness, max_corr_greeness = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"greeness"), ground_df,32618,"greeness")
# 1480
# rf1630


# In[ ]:


best_res_impervious, max_corr_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"impervious"), ground_df,32618,"impervious")
# 1480
# rf1530


# In[ ]:


best_res_landsat_lst, max_corr_landsat_lst = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsat_lst"), ground_df,32618,"landsat_lst")
# 880
# rf1330


# In[ ]:


best_res_mndwi, max_corr_mndwi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"mndwi"), ground_df,32618,"mndwi")
# 930
# rf530


# In[ ]:


best_res_ndbi, max_corr_ndbi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndbi"), ground_df,32618,"ndbi")
# 1280
# rf1330


# In[ ]:


best_res_ndli, max_corr_ndli = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndli"), ground_df,32618,"ndli")
# 30
# rf1330


# In[ ]:


best_res_ndvi, max_corr_ndvi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi"), ground_df,32618,"ndvi")
# 30
# rf1330


# In[ ]:


best_res_ndwi, max_corr_ndwi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndwi"), ground_df,32618,"ndwi")
# 30
# rf1630


# In[ ]:


best_res_surface_albedo, max_corr_surface_albedo = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surface_albedo"), ground_df,32618,"surface_albedo")
# 380
# rf880


# ### Optimal Resolution for Social Indicator

# In[ ]:


best_res_anh, max_corr_anh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"anh"), ground_df,32618,"anh")
# 930
# rf1580


# In[ ]:


best_res_bnh, max_corr_bnh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"bnh"), ground_df,32618,"bnh")
# 1480
# rf1630


# In[ ]:


best_res_hsp1, max_corr_hsp1 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"hsp1"), ground_df,32618,"hsp1")
# 1080
# rf1630


# In[ ]:


best_res_hunits, max_corr_hunits = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"hunits"), ground_df,32618,"hunits")
# 1480
# rf1580


# In[ ]:


best_res_mean_b_yea, max_corr_mean_b_yea = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"mean_b_yea"), ground_df,32618,"mean_b_yea")
# 180
# rf1630


# In[ ]:


best_res_mean_c_occ, max_corr_mean_c_occ = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"mean_c_occ"), ground_df,32618,"mean_c_occ")
# 1480
# rf1630


# In[ ]:


best_res_pop65pl, max_corr_pop65pl = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"pop65pl"), ground_df,32618,"pop65pl")
# 1480
# rf1630


# In[ ]:


best_res_twopinh, max_corr_twopinh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"twopinh"), ground_df,32618,"twopinh")
# 1230
# rf1430


# In[ ]:


best_res_vachus, max_corr_vachus = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"vachus"), ground_df,32618,"vachus")
# 180
# rf1080


# In[ ]:


best_res_wnh, max_corr_wnh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"wnh"), ground_df,32618,"wnh")
# 1480
# rf1630


# In[ ]:


best_res_build_cove, max_corr_build_cove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"build_cove"), ground_df,32618,"build_cove")
# 880
# rf1280


# In[ ]:


best_res_build_dens, max_corr_build_dens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"build_dens"), ground_df,32618,"build_dens")
# 30
# rf1630


# In[ ]:


best_res_mean_heigh, max_corr_mean_heigh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"mean_heigh"), ground_df,32618,"mean_heigh")
# 1480
# rf1630


# In[ ]:


best_res_popdensity, max_corr_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity"), ground_df,32618,"popdensity")
# 1480
# rf1630


# ### Optimal Resolution for Theoretical Interaction

# In[ ]:


best_res_ndvi_impervious, max_corr_ndvi_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi_impervious"), ground_df,32618,"ndvi_impervious")
# 1080
# rf30


# In[ ]:


best_res_ndvi_div_impervious, max_corr_ndvi_div_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi_div_impervious"), ground_df,32618,"ndvi_div_impervious")
# 1180
# rf1630


# In[ ]:


best_res_ndvi_buildcove, max_corr_ndvi_buildcove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi_buildcove"), ground_df,32618,"ndvi_buildcove")
# 30
# rf1630


# In[ ]:


best_res_ndvi_builddens, max_corr_ndvi_builddens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi_builddens"), ground_df,32618,"ndvi_builddens")
# 1430
# rf1030


# In[ ]:


best_res_greenessres_impervious, max_corr_greenessres_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"greenessres_impervious"), ground_df,32618,"greenessres_impervious")
# 30
# rf1630


# In[ ]:


best_res_ndvi_surfacealbedo, max_corr_ndvi_surfacealbedo = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"ndvi_surfacealbedo"), ground_df,32618,"ndvi_surfacealbedo")
# 30
# rf1530


# In[ ]:


best_res_landsatlst_buildcove, max_corr_landsatlst_buildcove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsatlst_buildcove"), ground_df,32618,"landsatlst_buildcove")
# 1230
# rf280


# In[ ]:


best_res_landsatlst_builddens, max_corr_landsatlst_builddens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsatlst_builddens"), ground_df,32618,"landsatlst_builddens")
# 1330
# rf980


# In[ ]:


best_res_landsatlst_meanheigh, max_corr_landsatlst_meanheigh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsatlst_meanheigh"), ground_df,32618,"landsatlst_meanheigh")
# 1180
# rf1430


# In[ ]:


best_res_landsatlst_impervious, max_corr_landsatlst_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsatlst_impervious"), ground_df,32618,"landsatlst_impervious")
# 1180
# rf1180


# In[ ]:


best_res_landsatlst_div_foresthight, max_corr_landsatlst_div_foresthight = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"landsatlst_div_foresthight"), ground_df,32618,"landsatlst_div_foresthight")
# 980
# rf1630


# In[ ]:


best_res_popdensity_impervious, max_corr_popdensity_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_impervious"), ground_df,32618,"popdensity_impervious")
# 1030
# rf1180


# In[ ]:


best_res_popdensity_buildcove, max_corr_popdensity_buildcove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_buildcove"), ground_df,32618,"popdensity_buildcove")
# 1180
# rf1230


# In[ ]:


best_res_popdensity_builddens, max_corr_popdensity_builddens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_builddens"), ground_df,32618,"popdensity_builddens")
# 1280
# rf1630


# In[ ]:


best_res_popdensity_meanheigh, max_corr_popdensity_meanheigh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_meanheigh"), ground_df,32618,"popdensity_meanheigh")
# 1130
# rf1130


# In[ ]:


best_res_popdensity_surfacealbedo, max_corr_popdensity_surfacealbedo = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_surfacealbedo"), ground_df,32618,"popdensity_surfacealbedo")
# 880
# rf1630


# In[ ]:


best_res_popdensity_landsatlst, max_corr_popdensity_landsatlst = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_landsatlst"), ground_df,32618,"popdensity_landsatlst")
# 980
# rf1180


# In[ ]:


best_res_popdensity_hsp1, max_corr_popdensity_hsp1 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_hsp1"), ground_df,32618,"popdensity_hsp1")
# 1180
# rf1080


# In[ ]:


best_res_popdensity_hunits, max_corr_popdensity_hunits = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"popdensity_hunits"), ground_df,32618,"popdensity_hunits")
# 1230
# rf1330


# In[ ]:


best_res_meanheigh_impervious, max_corr_meanheigh_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"meanheigh_impervious"), ground_df,32618,"meanheigh_impervious")
# 930
# rf1630


# In[ ]:


best_res_meanheigh_buildcove, max_corr_meanheigh_buildcove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"meanheigh_buildcove"), ground_df,32618,"meanheigh_buildcove")
# 1080
# rf1080


# In[ ]:


best_res_meanheigh_builddens, max_corr_meanheigh_builddens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"meanheigh_builddens"), ground_df,32618,"meanheigh_builddens")
# 1230
# rf1180


# In[ ]:


best_res_meanheigh_div_foresthight, max_corr_meanheigh_div_foresthight = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"meanheigh_div_foresthight"), ground_df,32618,"meanheigh_div_foresthight")
# 30
# rf1080


# In[ ]:


best_res_buildcove_meanb_yea, max_corr_buildcove_meanb_yea = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"buildcove_meanb_yea"), ground_df,32618,"buildcove_meanb_yea")
# 1230
# rf1630


# In[ ]:


best_res_buildcove_meanc_occ, max_corr_buildcove_meanc_occ = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"buildcove_meanc_occ"), ground_df,32618,"buildcove_meanc_occ")
# 1230
# rf680


# In[ ]:


best_res_impervious_hunits, max_corr_impervious_hunits = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"impervious_hunits"), ground_df,32618,"impervious_hunits")
# 1230
# rf1230


# In[ ]:


best_res_impervious_vachus, max_corr_impervious_vachus = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"impervious_vachus"), ground_df,32618,"impervious_vachus")
# 1480
# rf1480


# In[ ]:


best_res_surfacealbedo_impervious, max_corr_surfacealbedo_impervious = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_impervious"), ground_df,32618,"surfacealbedo_impervious")
# 330
# rf830


# In[ ]:


best_res_surfacealbedo_buildcove, max_corr_surfacealbedo_buildcove = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_buildcove"), ground_df,32618,"surfacealbedo_buildcove")
# 1280
# rf880


# In[ ]:


best_res_surfacealbedo_ndvi, max_corr_surfacealbedo_ndvi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_ndvi"), ground_df,32618,"surfacealbedo_ndvi")
# 30
# rf1530


# In[ ]:


best_res_surfacealbedo_landsatlst, max_corr_surfacealbedo_landsatlst = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_landsatlst"), ground_df,32618,"surfacealbedo_landsatlst")
# 280
# rf830


# In[ ]:


best_res_surfacealbedo_meanheigh, max_corr_surfacealbedo_meanheigh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_meanheigh"), ground_df,32618,"surfacealbedo_meanheigh")
# 880
# rf1380


# In[ ]:


best_res_surfacealbedo_popdensity, max_corr_surfacealbedo_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"surfacealbedo_popdensity"), ground_df,32618,"surfacealbedo_popdensity")
# 880
# rf1630


# In[ ]:


best_res_dem_forest_hight, max_corr_dem_forest_hight = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"dem_forest_hight"), ground_df,32618,"dem_forest_hight")


# In[ ]:


best_res_forest_hight_greeness, max_corr_forest_hight_greeness = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"forest_hight_greeness"), ground_df,32618,"forest_hight_greeness")


# In[ ]:


best_res_forest_hight_ndbi, max_corr_forest_hight_ndbi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path,"forest_hight_ndbi"), ground_df,32618,"forest_hight_ndbi")


# ### Optimal Resolution for Experimental Interaction

# In[ ]:


best_res_anh_build_dens, max_corr_anh_build_dens = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "anh_build_dens"), ground_df, 32618, "anh_build_dens")
# 1080
# rf1530


# In[ ]:


best_res_anh_popdensity, max_corr_anh_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "anh_popdensity"), ground_df, 32618, "anh_popdensity")
# 880
# rf1630


# In[ ]:


best_res_b12_ndbi, max_corr_b12_ndbi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "b12_ndbi"), ground_df, 32618, "b12_ndbi")
# 1380
# rf1330


# In[ ]:


best_res_b12_ndli, max_corr_b12_ndli = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "b12_ndli"), ground_df, 32618, "b12_ndli")
# 30
# rf1530


# In[ ]:


best_res_b4_ndli, max_corr_b4_ndli = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "b4_ndli"), ground_df, 32618, "b4_ndli")
# 30
# rf980


# In[ ]:


best_res_bnh_greeness, max_corr_bnh_greeness = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "bnh_greeness"), ground_df, 32618, "bnh_greeness")
# 1430
# rf1580


# In[ ]:


best_res_build_cove_hunits, max_corr_build_cove_hunits = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_cove_hunits"), ground_df, 32618, "build_cove_hunits")
# 1180
# rf1530


# In[ ]:


best_res_build_cove_pop65pl, max_corr_build_cove_pop65pl = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_cove_pop65pl"), ground_df, 32618, "build_cove_pop65pl")
# 480
# rf1480


# In[ ]:


best_res_build_dens_hunits, max_corr_build_dens_hunits = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_dens_hunits"), ground_df, 32618, "build_dens_hunits")
# 980
# rf1530


# In[ ]:


best_res_build_dens_pop65pl, max_corr_build_dens_pop65pl = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_dens_pop65pl"), ground_df, 32618, "build_dens_pop65pl")
# 1480
# rf1430


# In[ ]:


best_res_build_dens_twopinh, max_corr_build_dens_twopinh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_dens_twopinh"), ground_df, 32618, "build_dens_twopinh")
# 780
# rf1030


# In[ ]:


best_res_build_dens_vachus, max_corr_build_dens_vachus = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_dens_vachus"), ground_df, 32618, "build_dens_vachus")
# 630
# rf1630


# In[ ]:


best_res_build_dens_wnh, max_corr_build_dens_wnh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "build_dens_wnh"), ground_df, 32618, "build_dens_wnh")
# 1480
# rf1530


# In[ ]:


best_res_dem_mean_heigh, max_corr_dem_mean_heigh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_mean_heigh"), ground_df, 32618, "dem_mean_heigh")
# 1480
# rf1480


# In[ ]:


best_res_dem_ndli, max_corr_dem_ndli = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_ndli"), ground_df, 32618, "dem_ndli")
# 80
# rf1380


# In[ ]:


best_res_dem_ndvi, max_corr_dem_ndvi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_ndvi"), ground_df, 32618, "dem_ndvi")
# 80
# rf1430


# In[ ]:


best_res_dem_ndwi, max_corr_dem_ndwi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_ndwi"), ground_df, 32618, "dem_ndwi")
# 30
# rf1430


# In[ ]:


best_res_dem_popdensity, max_corr_dem_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_popdensity"), ground_df, 32618, "dem_popdensity")
# 930
# rf1630


# In[ ]:


best_res_dem_twopinh, max_corr_dem_twopinh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_twopinh"), ground_df, 32618, "dem_twopinh")
# 80
# rf1630


# In[ ]:


best_res_dem_wnh, max_corr_dem_wnh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "dem_wnh"), ground_df, 32618, "dem_wnh")
# 1480
# rf1630


# In[ ]:


best_res_greeness_hsp1, max_corr_greeness_hsp1 = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "greeness_hsp1"), ground_df, 32618, "greeness_hsp1")
# 1430
# rf1280


# In[ ]:


best_res_hunits_popdensity, max_corr_hunits_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "hunits_popdensity"), ground_df, 32618, "hunits_popdensity")
# 1480
# rf1630


# In[ ]:


best_res_mean_heigh_twopinh, max_corr_mean_heigh_twopinh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "mean_heigh_twopinh"), ground_df, 32618, "mean_heigh_twopinh")
# 1280
# rf1330


# In[ ]:


best_res_mndwi_ndbi, max_corr_mndwi_ndbi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "mndwi_ndbi"), ground_df, 32618, "mndwi_ndbi")
# 30
# rf130


# In[ ]:


best_res_ndli_ndvi, max_corr_ndli_ndvi = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "ndli_ndvi"), ground_df, 32618, "ndli_ndvi")
# 1480
# rf1630


# In[ ]:


best_res_pop65pl_popdensity, max_corr_pop65pl_popdensity = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "pop65pl_popdensity"), ground_df, 32618, "pop65pl_popdensity")
# 1280
# rf1630


# In[ ]:


best_res_popdensity_twopinh, max_corr_popdensity_twopinh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "popdensity_twopinh"), ground_df, 32618, "popdensity_twopinh")
# 1380
# rf1380


# In[ ]:


best_res_popdensity_wnh, max_corr_popdensity_wnh = find_best_resolution_correlation(os.path.join(config.resampled_rasters_path, "popdensity_wnh"), ground_df, 32618, "popdensity_wnh")
# 980
# rf1180


# In[4]:


get_ipython().system('jupyter nbconvert --to script preprocess.ipynb')

