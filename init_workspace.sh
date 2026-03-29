#!/bin/bash

mkdir -p workspace
cd workspace

mkdir -p datasets
cd datasets

cd ..
mkdir -p models
cd models

echo -e "For cell lines and assays of interest, please follow the following steps for preparing trained model checkpoints and datasets for use:" 
echo -e "1. Download the dataset and model from Zenodo as outlined in the README.md."
echo -e "2. Unzip the downloaded file and place the model checkpoints in workspace/models."
echo -e "3. Unzip the downloaded file and place the datasets in workspace/datasets."