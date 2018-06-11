# MultitaskAIS

This folder contains the TensorFlow implementation of the model proposed in

"paper name"
"paper arxiv link"

All the codes related to the Embedding block are adapted from the source code of Filtering Variational Objectives:
https://github.com/tensorflow/models/tree/master/research/fivo

#### Directory Structure
The elements of the code are organized as follows:

```
multitaskAIS.py                   # cript to train the Embedding layer
runners.py                        # graph construction code for training and evaluation
bounds.py                         # code for computing each bound
data
├── datasets.py                   # readers for AIS dataset
├── calculate_AIS_mean.py         # calculates the mean of AIS "four-hot" vectors
├── dataset_preprocessing.py      # preprocesses the AIS datasets
└── csv2pkl.py                    # loads AIS data from *.csv files 
models
└── vrnn.py                       # variational RNN implementation

```

### Datasets:

The MarineC dataset is availble at [https://marinecadastre.gov/ais/]

The Brittany dataset is provided by Collecte Localisation Satellites


#### Preprocess the Data

Converting to csv:
* MarineC dataset: we use GDIS (http://gdis.seul.org/) to convert the original metadata format to csv files.
* Brittany dataset: we use libais (https://github.com/schwehr/libais) to decode raw AIS messages to csv files.

`csv2pkl.py` then loads the data from csv files, selects AIS messages in the pre-defined ROI then saves them as pickle format.

Preprocessing steps: the data then processed as discribed in the paper by `dataset_preprocessing.py`

### Training the Embedding layer

First we must train the Embedding layer:
```
python multitaskAIS.py \
  --mode=train \
  --logdir=/tmp/multitaskAIS \
  --bound=elbo \
  --summarize_every=100 \
  --latent_size=100 \
  --batch_size=50 \
  --num_samples=16 \
  --learning_rate=0.0003 \
```

### Running task-specific submodels

After the Embedding layer is trained, we can run task-specific blocks.
To avoid re-caculating the $log[p(x_t|x_{1..t-1},x_{1..t-1})]$, we 
To optimize the calculatetion time, we 

#### Trajectory reconstruction

We tested the Trajectory re


### Acknowledgement

We would like to thank Tensorflow team and OpenStreetmap for the open-source code and data.


### Contact

This code is a raw version of MultitaskAIS. The code will be maintained and obtimized.
For any questions/issues, please contact Duong NGUYEN via van.nguyen1@imt-atlantique.fr
