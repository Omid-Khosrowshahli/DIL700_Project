This project aims to define and train two models for pixel-level semantic segmentation tasks on 
the Marine Debris Archive (MARIDA) with an effort to enhance the results from the baseline algorithms.
A U-net structure and a XGBoost algorithm.

The original work from the creators of the MARIDA dataset can be found at:
> Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K (2022) MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLoS ONE 17(1): e0262247. https://doi.org/10.1371/journal.pone.0262247

The repository of the original work can be found at: https://github.com/marine-debris/marine-debris.github.io.

In order to download MARIDA go to https://doi.org/10.5281/zenodo.5151941.

### Structure
To train and evaluate the models, the folder structure needs to be the same as the structure of the repository.
The dataset should be downloaded and extracted in the `data/` folder.

### ELU-Unet
The code for both modified U-net model with ELU activation function and the U-net structure with a pretrained Res-Net bottleneck can be found in `semantic_segmentation/ELU-Unet/` folder. The reported results in the report of the project are related to ELU-Unet model due to its better performance.
For training the model and evaluating it on the validation set run:

```bash
cd semantic_segmentation/ELU-Unet
python train.py
```
And run the following to evaluater the model on the test set:

```bash
cd semantic_segmentation/ELU-Unet
python evaluation.py
```

### XGBoost
First run the following lines to create the necessary files related to Spectral Signatures, Spectral Indices and texture features.

This code creates `dataset.h5`:

```bash
cd semantic_segmentation/XGBoost
python engineering_patches.py
```
To create `dataset_si.h5` run:

```bash
python utils/spectral_extraction.py --type indices
```
For the stacked GLCM patches run:

```bash
python engineering_patches.py --type texture
```
And to produce the `dataset_glcm.h5` run:

```bash
python utils/spectral_extraction.py --type texture
```

For training and evaluating the XGBoost model, run:

```bash
cd semantic_segmentation\XGBoost
python train_eval.py
```

### Data Visualization
The codes to visualize the data are stored in the visualization folder. To inspect any patch or shapefile from the dataset, place the necessary patch or shapefile in the folder and change the file names in the python codes accordingly.