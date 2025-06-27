import os
import pandas as pd
import json

class Path_config():
    def __init__(self):
        # Main root directory
        self.root = rf"D:\EY_2025\model"
        # Main folder paths
        self.data_dir = os.path.join(self.root, "data")
        self.raw_data_path = os.path.join(self.data_dir, "raw")
        self.processed_data_path = os.path.join(self.data_dir, "processed")
        self.reprojected_rasters_path = os.path.join(self.processed_data_path, "reproject_rasters")
        self.resampled_rasters_path = os.path.join(self.processed_data_path, "resampled_rasters")
        # Input file paths
        self.uhi_path = os.path.join(self.raw_data_path, "uhi_data", "Training_data_uhi_index_2025-02-18.csv")
        self.env_path = os.path.join(self.raw_data_path, "environmental_data")
        self.censusblock_path = os.path.join(self.raw_data_path, "NYC_censusblock_nonspatialfactors_model.shp")
        self.test_csv_path = os.path.join(self.raw_data_path, "uhi_data", "Submission_template_UHI2025-v2.csv")
        self.transform_info_path = os.path.join(self.processed_data_path, "yeojohnson_transform_info.csv")
        # Path for selected feature list
        self.features_path = os.path.join(self.processed_data_path, "selected_features.csv")
        
        # Environmental variable file names
        self.env_files = [
            # Environmental Indicator
            "B1.tif", "B2.tif", "B3.tif", "B4.tif", "B5.tif", "B6.tif", "B7.tif", "B8.tif","B8A.tif",
            "B9.tif", "B10.tif", "B11.tif", "B12.tif", "DEM.tif", "Forest_hight.tif", "Greeness.tif",
            "impervious.tif", "landsat_LST.tif", "MNDWI.tif", "NDBI.tif", "NDLI.tif", "NDVI.tif", "NDWI.tif", "Surface_Albedo.tif",
            # Social Indicator
            "ANH.tif", "BNH.tif", "Hsp1.tif", "HUnits.tif", "MEAN_B_Yea.tif", "MEAN_C_Occ.tif", "Pop65pl.tif", "TwoPINH.tif",
            "VacHUs.tif", "WNH.tif", "MEAN_heigh.tif", "PopDensity.tif","Build_cove.tif", "Build_Dens.tif",
            # Theoretical Interaction
            "ndvi_impervious.tif","ndvi_builddens.tif","greenessres_impervious.tif",
            "ndvi_surfacealbedo.tif","landsatlst_meanheigh.tif","landsatlst_impervious.tif",
            "popdensity_impervious.tif","popdensity_buildcove.tif","popdensity_builddens.tif","popdensity_meanheigh.tif",
            "popdensity_surfacealbedo.tif","popdensity_landsatlst.tif","popdensity_hsp1.tif","popdensity_hunits.tif","meanheigh_impervious.tif",
            "meanheigh_builddens.tif","buildcove_meanb_yea.tif","buildcove_meanc_occ.tif",
            "impervious_hunits.tif","impervious_vachus.tif","surfacealbedo_impervious.tif","surfacealbedo_buildcove.tif","surfacealbedo_ndvi.tif",
            "surfacealbedo_landsatlst.tif","surfacealbedo_meanheigh.tif","surfacealbedo_popdensity.tif","landsatlst_buildcove.tif","landsatlst_builddens.tif"
            ,"ndvi_div_impervious.tif","landsatlst_div_foresthight.tif","meanheigh_div_foresthight.tif","ndvi_buildcove.tif","meanheigh_buildcove.tif",
            "dem_forest_hight.tif", "forest_hight_greeness.tif", "forest_hight_ndbi.tif",
            # Experimental Interaction (Correlation > 0.2 and Non-collinearity)
            "anh_build_dens.tif", "anh_popdensity.tif", "b12_ndbi.tif", "b12_ndli.tif", "b4_ndli.tif", "bnh_greeness.tif", "build_cove_hunits.tif", 
            "build_cove_pop65pl.tif", "build_dens_hunits.tif", "build_dens_pop65pl.tif", "build_dens_twopinh.tif", "build_dens_vachus.tif", "build_dens_wnh.tif",
            "dem_mean_heigh.tif", "dem_ndli.tif", "dem_ndvi.tif", "dem_ndwi.tif", "dem_popdensity.tif", "dem_twopinh.tif", "dem_wnh.tif",
            "greeness_hsp1.tif", "hunits_popdensity.tif", "mean_heigh_twopinh.tif", "mndwi_ndbi.tif", "ndli_ndvi.tif", 
            "pop65pl_popdensity.tif", "popdensity_twopinh.tif", "popdensity_wnh.tif"
        ]
        
        self.rasters_res = {
            "b1": 780, "b2": 780, "b3": 780, "b4": 780, "b5": 780,
            "b6": 380, "b7": 480, "b8": 780, "b8a": 780, "b9": 730,
            "b10": 1630, "b11": 880, "b12": 780, "dem": 580, "forest_hight": 180, "greeness": 1630,
            "impervious": 1530,"landsat_lst": 1330,"mndwi": 530, "ndbi": 1330, "ndli": 1330, "ndvi": 1330, "ndwi": 1630, "surface_albedo": 880,
            "anh": 1580, "bnh": 1630, "hsp1": 1630, "hunits": 1580, "mean_b_yea": 1630, "mean_c_occ": 1630, "pop65pl": 1630,
            "twopinh": 1430, "vachus": 1080, "wnh": 1630, "build_cove": 1280, "build_dens": 1630, "mean_heigh": 1630, "popdensity": 1630,
            # interaction
            "ndvi_impervious": 30,"ndvi_div_impervious":1630,"ndvi_buildcove":1630,"ndvi_builddens":1030,"greenessres_impervious":1630,
            "ndvi_surfacealbedo":1530,"landsatlst_buildcove":280,"landsatlst_builddens":980,"landsatlst_meanheigh":1430,"landsatlst_impervious":1180,
            "landsatlst_div_foresthight":1630,"popdensity_impervious":1180,"popdensity_buildcove":1230,"popdensity_builddens":1630,"popdensity_meanheigh":1130,
            "popdensity_surfacealbedo":1630,"popdensity_landsatlst":1180,"popdensity_hsp1":1080,"popdensity_hunits":1330,"meanheigh_impervious":1630,
            "meanheigh_buildcove":1080,"meanheigh_builddens":1180,"meanheigh_div_foresthight":1580,"buildcove_meanb_yea":1630,"buildcove_meanc_occ":680,
            "impervious_hunits":1230,"impervious_vachus":1480,"surfacealbedo_impervious":830,"surfacealbedo_buildcove":880,"surfacealbedo_ndvi":1530,
            "surfacealbedo_landsatlst":830,"surfacealbedo_meanheigh":1380,"surfacealbedo_popdensity":1630,
            "dem_forest_hight":1130, "forest_hight_greeness":1630, "forest_hight_ndbi":1380,
            # interaction add
            "anh_build_dens":1280, "anh_popdensity":1630, "b12_ndbi":1330, "b12_ndli":1530, "b4_ndli":980, "bnh_greeness":1630, 
            "build_cove_hunits":1530, "build_cove_pop65pl":1480, "build_dens_hunits":1530, "build_dens_pop65pl":1630, "build_dens_twopinh":1030, 
            "build_dens_vachus":1630, "build_dens_wnh":1280,"dem_mean_heigh":1480, "dem_ndli":1380, "dem_ndvi":1430, "dem_ndwi":1430, 
            "dem_popdensity":1630, "dem_twopinh":1630, "dem_wnh":1630,  
            "greeness_hsp1":1280, "hunits_popdensity":1630, "mean_heigh_twopinh":1330, "mndwi_ndbi":130, "ndli_ndvi":1630, "pop65pl_popdensity":1630, 
            "popdensity_twopinh":1380, "popdensity_wnh":1180
        }       
        
        # Generate variable names from file names (remove extension and convert to lowercase)
        self.env_names = [file_name.lower().split('.')[0] for file_name in self.env_files]
        # Generate a dictionary of environmental variable file paths
        self.file_paths = {}
        for name, file in zip(self.env_names, self.env_files):
            self.file_paths[name] = os.path.join(self.env_path, file)
        self.file_paths_list = list(self.file_paths.values())
        # Output file paths
        self.combined_data_path = os.path.join(self.processed_data_path, "combined_data.csv")
        self.training_data_path = os.path.join(self.processed_data_path, "training_data.csv")
        self.training_data_reduce_path = os.path.join(self.processed_data_path, "training_data_reduce.csv")
        self.reduce_x_path = os.path.join(self.processed_data_path, "reduce_x.csv")
        self.submit_data_path = os.path.join(self.processed_data_path, "submit_data.csv")
        # Model storage path
        self.model_path = os.path.join(self.root, "models")
        self.figures_path = os.path.join(self.root, "reports", "figures")
        
        # Other paths
        self.selected_features_path = os.path.join(self.processed_data_path, "selected_features.csv")
        
        self.cols_to_keep = pd.read_csv(self.selected_features_path)
        self.cols_to_keep = self.cols_to_keep['Feature'].tolist()
        
        self.rename_list = dict(zip(self.cols_to_keep, [
            "UHI Index",
            "Building density\n X W",
            "A",
            "Mean building height\n X Impervious",
            "NDLI X NDVI",
            "Mean building Height\n X Building coverage",
            "NDVI"
        ]))
        
        with open(os.path.join(self.data_dir, "lambda_dict.json")) as f:
            self.lambda_dict = json.load(f)
                

        
path_config = Path_config()