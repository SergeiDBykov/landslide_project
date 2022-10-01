# landslide_project

TUM.ai Makeathon Fall 2022 contribution for the Environment track.

## Model Part

* `data` - a folder with data used for building a model. This is [NASA's Landslide catalog](https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Not-updated-/h9d8-neg4), can be downloaded at [Kaggle](https://www.kaggle.com/datasets/sathyanarayanrao89/global-landslide-catalog?sort=votes).
* `model` - a folder in which models are stored, and some plots from it.
* `eda.ipynb` - exploration of the dataset and first attempts to build a spatial-temporal model. 
* `data_generation.ipynb` - the notebook in which we use `weather api` to gather rainfall data to extend the landslide data
* `fit.py` - script in which the model are trained and stored
* `predict.py` (moved to `backend/model`) - script for model inference.
* `predict_example.ipynb` - notebook with example of a prediction. 
