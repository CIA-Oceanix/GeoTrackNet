# MultitaskAIS

TensorFlow implementation of the model proposed in "A Multi-Task Deep Learning Architecture for Maritime Surveillance Using AIS Data Streams" (https://ieeexplore.ieee.org/abstract/document/8631498) and "GeoTrackNet—A Maritime Anomaly Detector using Probabilistic Neural Network Representation of AIS Tracks and A Contrario Detection" (https://arxiv.org/abs/1912.00682).

All the codes related to the Embedding block are adapted from the source code of Filtering Variational Objectives:
https://github.com/tensorflow/models/tree/master/research/fivo


#### Directory Structure
The elements of the code are organized as follows:

```
multitaskAIS.py                   # script to run the model (except the A contrario detection).
runners.py                        # graph construction code for training and evaluation.
bounds.py                         # code for computing each bound.
contrario_kde.py                  # script to run the A contrario detection.
contrario_utils.py
distribution_utils.py
nested_utils.py
utils.py
data
├── datasets.py                   # reader pipelines.
├── calculate_AIS_mean.py         # calculates the mean of the AIS "four-hot" vectors.
├── dataset_preprocessing.py      # preprocesses the AIS datasets.
└── csv2pkl.py                    # parse raw AIS messages from aivdm format to csv files.
└── csv2pkl.py                    # loads AIS data from *.csv files.
models
└── vrnn.py                       # VRNN implementation.
chkpt
└── ...                           # directory to keep checkpoints and summaries in.
results
└── ...                           # directory to save results to.
```

#### Requirements: 
See requirements.yml

### Datasets:

The MarineC dataset is provided by MarineCadastre.gov, Bureau of Ocean Energy Management, and National Oceanic and Atmospheric Administration, (marinecadastre.gov), and availble at (https://marinecadastre.gov/ais/)

The Brittany dataset is provided by CLS-Collecte Localisation Satellites (https://www.cls.fr/en/) and Erwan Guegueniat, comprises AIS messages captured by a coastal receiving station in Ushant, from 07/2011 to 07/2019. We provide here a set of processed AIS messages (data/ct_2017010203_10_20.zip) on which readers can re-produce the results in the paper GeoTrackNet. This set comprises dynamic information of AIS tracks (LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI) of cargo and tanker vessels from 01/2017 to 03/2017, downsampled to a resolution of 5 minutes. For the full Brittany dataset, please contact CLS (G. Hajduch, ghajduch@groupcls.com).

#### Preprocess the Data

Converting to csv:
* MarineC dataset: we use QGIS (https://qgis.org/en/site/) to convert the original metadata format to csv files.
* Brittany dataset: we use libais (https://github.com/schwehr/libais) to parse raw AIS messages to csv files (see avidm_decoder.py).

`csv2pkl.py` then loads the data from csv files, selects AIS messages in the pre-defined ROI then saves them as pickle format.

Preprocessing steps: the data are processed as described in the paper by `dataset_preprocessing.py`.

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

A model trained on the dataset comprising AIS messages of cargo and tanker vessels, from January 01 to March 10, 2017 can be found at `chkpt/elbo-ct_2017010203_10_20_train.pkl-data_dim-602-latent_size-100-batch_size-50.zip`.

### Running task-specific submodels

After the Embedding layer is trained, we can run task-specific blocks.


#### save_outcomes
To avoid re-caculating $log[p(x_t|h_t)]$ for each task, we calculate them once and save the results as a .pkl file. 
```
python multitaskAIS.py \
  --mode=save_outcomes \
  --logdir=./chkpt \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 
Similarly for the test set (```testset_name=ct_2017010203_10_20/ct_2017010203_10_20_test.pkl```).

#### log_density
*log_density* calculates the distribution of $log[p(x_t|h_t)]$ in each small cells of the ROI.
```
python multitaskAIS.py \
  --mode=log_density \
  --logdir=./chkpt \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 

#### contrario detection
*contrario_kde.py* performs the a contrario detection and plots the results.
```
python contrario_kde.py \
``` 
#### traj_reconstruction
*traj_reconstruction* performs the trajectory reconstruction.

*Note: this task works only in busy traffic regions. Since our main focus is anomaly detection, we put little effort into this task.*

```
python multitaskAIS.py \
  --mode=traj_reconstruction \
  --logdir=./chkpt \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --bound=elbo \
  --latent_size=100 \
  --batch_size=1 \
  --num_samples=16 \
``` 


### Acknowledgement

We would like to thank MarineCadastre, CLS and Erwan Guegueniat, Kurt Schwehr, Tensorflow team, QGIS and OpenStreetmap for the data and the open-source codes.

We would also like to thank Jetze Schuurmans for helping convert the code from Python2 to Python3.

### Contact
For any questions, please open an issue.
