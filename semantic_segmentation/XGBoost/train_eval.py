# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Further Modifications: Omid Khosrowshahli
Email: khosrowshahli.omid@gmail.com
Python Version: 3.7.10
Description: train_eval.py includes the training and evaluation process for the
             pixel-level semantic segmentation with XGBoost.
'''

import os
import ast
import sys
import time
import json
import random
import logging
import rasterio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
from XGBoost import xgb_random_search, bands_mean

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import confusion_matrix
from assets import conf_mapping, rf_features, cat_mapping_vec

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

random.seed(0)
np.random.seed(0)

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(
    filename=os.path.join(root_path, 'logs', 'evaluation_xgb.log'), 
    filemode='a',
    level=logging.INFO, 
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.info('*' * 10)

conf_mat_labels = ['Clauds', 'DenS', 'Foam', 'MD', 'MWater', 'NatM', 'SLWater', 'SWater', 'Ship', 'SpS', 'TWater']


###############################################################
# Training                                                    #
###############################################################

def main(options):
    
    # Load Spectral Signatures, Spectral Indices, and GLCM texture features
    hdf_ss = pd.HDFStore(os.path.join(options['path'], 'dataset.h5'), mode='r')
    df_train_ss = hdf_ss.select('train')
    df_val_ss = hdf_ss.select('val')
    df_test_ss = hdf_ss.select('test')
    hdf_ss.close()
    
    hdf_si = pd.HDFStore(os.path.join(options['path'], 'dataset_si.h5'), mode='r')
    df_train_si = hdf_si.select('train')
    df_val_si = hdf_si.select('val')
    df_test_si = hdf_si.select('test')
    hdf_si.close()
    
    hdf_glcm = pd.HDFStore(os.path.join(options['path'], 'dataset_glcm.h5'), mode='r')
    df_train_glcm = hdf_glcm.select('train')
    df_val_glcm = hdf_glcm.select('val')
    df_test_glcm = hdf_glcm.select('test')
    hdf_glcm.close()
    
    # Merge features
    df_train = df_train_ss.merge(df_train_si, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    df_val = df_val_ss.merge(df_val_si, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    df_test = df_test_ss.merge(df_test_si, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    
    df_train = df_train.merge(df_train_glcm, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    df_val = df_val.merge(df_val_glcm, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    df_test = df_test.merge(df_test_glcm, on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'])
    
    # Calculate weights for each sample based on Confidence Level
    df_train['Weight'] = 1 / df_train['Confidence'].apply(lambda x: conf_mapping[x])
    
    # Aggregate selected classes into the Water Superclass
    for agg_class in options['agg_to_water']:
        df_train.loc[df_train['Class'] == agg_class, 'Class'] = 'Marine Water'
        df_val.loc[df_val['Class'] == agg_class, 'Class'] = 'Marine Water'
        df_test.loc[df_test['Class'] == agg_class, 'Class'] = 'Marine Water'
    
    # Prepare features and labels
    X_train = df_train[rf_features].values
    y_train = label_encoder.fit_transform(df_train['Class'].values)
    weight_train = df_train['Weight'].values
    
    if options['eval_set'] == 'test':
        X_test = df_test[rf_features].values
        y_test = label_encoder.transform(df_test['Class'].values)
    elif options['eval_set'] == 'val':
        X_test = df_val[rf_features].values
        y_test = label_encoder.transform(df_val['Class'].values)
    else:
        raise ValueError("Invalid eval_set option. Use 'val' or 'test'.")

    # Print and log class mapping
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nClass Mapping:")
    for class_name, encoded_value in class_mapping.items():
        print(f"{class_name}: {encoded_value}")

    logging.info(f"Class Mapping: {class_mapping}")
    
    print(f"Number of Input Features: {X_train.shape[1]}")
    print(f"Train Samples: {X_train.shape[0]}")
    print(f"Test Samples: {X_test.shape[0]}")
    
    logging.info(f"Number of Input Features: {X_train.shape[1]}")
    logging.info(f"Train Samples: {X_train.shape[0]}")
    logging.info(f"Test Samples: {X_test.shape[0]}")
        
    # Training
    print("Started training")
    logging.info("Started training")
    
    start_time = time.time()
    xgb_random_search.fit(X_train, y_train, **dict(xgb__sample_weight=weight_train))
    
    # Save model
    best_model = xgb_random_search.best_estimator_
    cl_path = os.path.join(up(os.path.abspath(__file__)), 'xgb_classifier_best_model.joblib')
    dump(best_model, cl_path)
    print(f"Best classifier is saved at: {cl_path}")
    logging.info(f"Best classifier is saved at: {cl_path}")
    
    best_params = xgb_random_search.best_params_
    print(f"Best Hyperparameters: {best_params}")
    logging.info(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}")

    # Evaluation
    print("\nEvaluating XGBoost on", options['eval_set'], "Set")
    predicted_classes = label_encoder.inverse_transform(best_model.predict(X_test).astype(int))
    true_classes = label_encoder.inverse_transform(y_test)
    
    conf_mat = confusion_matrix(true_classes, predicted_classes, label_encoder.classes_)
    logging.info(f"Confusion Matrix:\n{conf_mat.to_string()}")
    print("Confusion Matrix:\n", conf_mat.to_string())

    # Plot confusion matrix
    conf_mat_display = ConfusionMatrixDisplay.from_predictions(true_classes, predicted_classes, normalize="true", display_labels=conf_mat_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    conf_mat_display.plot(values_format='.2f', ax=ax)

    plt.xticks(rotation=45, ha="right")

    for text in ax.texts:
        text.set_fontsize(12)
        text.set_text(f"{float(text.get_text()) * 100:.0f}%")

    # Save the figure
    conf_matrix_path = os.path.join(root_path, 'logs', 'confusion_matrix.png')
    plt.savefig(conf_matrix_path, bbox_inches='tight', dpi=300)
    plt.close()

    if options['predict_masks']:
    
        path = os.path.join(options['path'], 'patches')
        ROIs = np.genfromtxt(os.path.join(options['path'], 'splits', 'test_X.txt'),dtype='str')
        
        impute_nan = np.tile(bands_mean, (256,256,1))
                    
        for roi in tqdm(ROIs):
        
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])             # Get Folder Name
            roi_name = '_'.join(['S2'] + roi.split('_'))                    # Get File Name
            roi_file = os.path.join(path, roi_folder,roi_name + '.tif')     # Get File path
        
            os.makedirs(options['gen_masks_path'], exist_ok=True)
        
            output_image = os.path.join(options['gen_masks_path'], os.path.basename(roi_file).split('.tif')[0] + '_rf.tif')
            
            # Load the image patch and metadata
            with rasterio.open(roi_file, mode ='r') as src:
                tags = src.tags().copy()
                meta = src.meta
                image = src.read()
                image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                dtype = src.read(1).dtype
        
            # Update meta to reflect the number of layers
            meta.update(count = 1)
            
            # Preprocessing
            # Fill image nan with mean
            nan_mask = np.isnan(image)
            image[nan_mask] = impute_nan[nan_mask]
            
            sz1 = image.shape[0]
            sz2 = image.shape[1]
            image_features = np.reshape(image, (sz1*sz2, -1))
            
            # Load Indices
            si_filename = os.path.join(options['path'], 'indices', roi_folder,roi_name + '_si.tif')
            with rasterio.open(si_filename, mode ='r') as src:
                image_si = src.read()
                image_si = np.moveaxis(image_si, (0, 1, 2), (2, 0, 1))
                
                si_sz1 = image_si.shape[0]
                si_sz2 = image_si.shape[1]
                si_image_features = np.reshape(image_si, (si_sz1*si_sz2, -1))

                si_image_features = np.nan_to_num(si_image_features)
            
            # Load Texture
            glcm_filename = os.path.join(options['path'], 'texture', roi_folder,roi_name + '_glcm.tif')
            with rasterio.open(glcm_filename, mode ='r') as src:
                image_glcm = src.read()
                image_glcm = np.moveaxis(image_glcm, (0, 1, 2), (2, 0, 1))
                
                glcm_sz1 = image_glcm.shape[0]
                glcm_sz2 = image_glcm.shape[1]
                glcm_image_features = np.reshape(image_glcm, (glcm_sz1*glcm_sz2, -1))

                glcm_image_features = np.nan_to_num(glcm_image_features)
                
            # Concatenate all features
            image_features = np.concatenate([image_features, si_image_features, glcm_image_features], axis=1)  
        
            # Write it
            with rasterio.open(output_image, 'w', **meta) as dst:
                
                # use classifier to predict labels for the whole image
                predictions = xgb_random_search.predict(image_features)  
    
                predicted_labels = np.reshape(predictions, (sz1,sz2))
    
                class_ind = cat_mapping_vec(predicted_labels).astype(dtype).copy()
                dst.write_band(1, class_ind) # In order to be in the same dtype
    
                dst.update_tags(**tags)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', default=os.path.join(root_path, 'data'), help='Path to dataset')
    parser.add_argument('--eval_set', default='test', type=str, help="Set for evaluation: 'val' or 'test'")
    parser.add_argument('--predict_masks', default=True, type=bool, help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_xgb'), help='Path to store predictions')

    parser.add_argument('--agg_to_water', default='["Mixed Water", "Wakes", "Cloud Shadows", "Waves"]', type=str, help='Classes to merge into Marine Water')

    args = parser.parse_args()
    options = vars(args)  # Convert to dictionary
    
    # Parse aggregation classes correctly
    agg_to_water = ast.literal_eval(options['agg_to_water'])
    options['agg_to_water'] = agg_to_water if isinstance(agg_to_water, list) else [agg_to_water]

    logging.info('Parsed input parameters:')
    logging.info(json.dumps(options, indent=2))

    main(options)
