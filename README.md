# MultitaskAIS

TensorFlow implementation of the model proposed in "Multi-task Learning for Maritime Traffic Surveillance from AIS Data Streams" (https://arxiv.org/abs/1806.03972).

All the codes related to the Embedding block are adapted from the source code of Filtering Variational Objectives:
https://github.com/tensorflow/models/tree/master/research/fivo

#### Directory Structure
The elements of the code are organized as follows:

```
multitaskAIS.py                   # script to train the Embedding layer
runners.py                        # graph construction code for training and evaluation
bounds.py                         # code for computing each bound
eval_multitaskAIS.py              # script to run task-specific layers (except the Contrario detection)
contrario.py                      # script to run the Contrario detection block
contrario_utils.py
distribution_utils.py
nested_utils.py
utils.py
get_coastline_streetmap.py        # script to download the coastline shapefile
data
├── datasets.py                   # readers for AIS dataset
├── calculate_AIS_mean.py         # calculates the mean of AIS "four-hot" vectors
├── dataset_preprocessing.py      # preprocesses the AIS datasets
└── csv2pkl.py                    # loads AIS data from *.csv files 
models
└── vrnn.py                       # variational RNN implementation
chkpt
└── ...                           # directory to keep checkpoints and summaries in
results
└── ...                           # directory to save outcomes
```

### Datasets:

The MarineC dataset is provided by MarineCadastre.gov, Bureau of Ocean Energy Management, and National Oceanic and Atmospheric Administration, (marinecadastre.gov), and availble at (https://marinecadastre.gov/ais/)

The Brittany dataset is provided by CLS-Collecte Localisation Satellites (https://www.cls.fr/en/) and Erwan Guegueniat, contains AIS messages captured by a coastal receiving station in Ushant, from 07/2011 to 01/2018. We provide here a set of preprocessed AIS messages (data/dataset8.zip) on which readers can re-produce the results in the paper. This set contains dynamic information of AIS tracks (LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI) from 01/2017 to 03/2017, downsampled to a resolution of 5 minutes. For the full Brittany dataset, please contact CLS (G.Hajduch, ghajduch@groupcls.com).  


#### Preprocess the Data

Converting to csv:
* MarineC dataset: we use QGIS (https://qgis.org/en/site/) to convert the original metadata format to csv files.
* Brittany dataset: we use libais (https://github.com/schwehr/libais) to decode raw AIS messages to csv files.

`csv2pkl.py` then loads the data from csv files, selects AIS messages in the pre-defined ROI then saves them as pickle format.

Preprocessing steps: the data then processed as discribed in the paper by `dataset_preprocessing.py`

### Training the Embedding layer

First we must train the Embedding layer:
```
python multitaskAIS.py \
  --mode=train \
  --logdir=./chkpt \
  --bound=elbo \
  --summarize_every=100 \
  --latent_size=100 \
  --batch_size=50 \
  --num_samples=16 \
  --learning_rate=0.0003 \
```

### Running task-specific submodels

After the Embedding layer is trained, we can run task-specific blocks.

#### save_outcomes
To avoid re-caculating the $log[p(x_t|x_{1..t-1},x_{1..t-1})]$ for each tasks, we calculate them once and save as .pkl file. 
```
python eval_multitaskAIS.py \
  --mode=save_outcomes \
  --logdir=./chkpt \
  --trainingset_name=dataset8/dataset8_train.pkl \
  --testset_name=dataset8/dataset8_valid.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 
Similarly for the test set (```testset_name=dataset8/dataset8_valid.pkl```).

#### log_density
*log_density* calculates the distribution of $log[p(x_t|x_{1..t-1},x_{1..t-1})]$ in each small cells of the ROI.
```
python eval_multitaskAIS.py \
  --mode=save_outcomes \
  --logdir=./chkpt \
  --trainingset_name=dataset8/dataset8_train.pkl \
  --testset_name=dataset8/dataset8_valid.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 

#### contrario detection
*contrario.py* performs the contrario detection and plots the results.
```
python contrario.py \
``` 
#### traj_reconstruction
*traj_reconstruction* performs the trajectory reconstruction.
```
python eval_multitaskAIS.py \
  --mode=traj_reconstruction \
  --logdir=./chkpt \
  --trainingset_name=dataset8/dataset8_train.pkl \
  --testset_name=dataset8/dataset8_test.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 


### Acknowledgement

We would like to thank MarineCadastre, CLS and Erwan Guegueniat, Tensorflow team and OpenStreetmap for the data and the open-source code.


### Contact

This code is a raw version of MultitaskAIS. We are sorry for not providing a clean version of the code, it is being optimized.
For any questions/issues, please contact Duong NGUYEN via van.nguyen1@imt-atlantique.fr
