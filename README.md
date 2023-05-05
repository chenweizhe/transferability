# A Study of Transferability of Deep Learning Models for Network Intrusion Detection
Transfer Learning for Network Traffic Anomaly Detection using Centralized ML

Intrusion detection systems often employ machine learning for network anomaly detection. These ML algorithms are trained to detect a set of common/well known attacks. However, these models may be equipped to detect a wider range of attacks than just the ones they are trained to detect. This can be determined using transfer learning. With the rise in the number of zero day attacks, there are no guarantees on what kind of an attack a network might be susceptible to. Transfer learning enables us to understand the full range of attacks that the ML algorithm is robust against and may potentially be able to detect zero day attacks too. The code for this work is available in this repository.

# Environment

conda env create -f environment.yml

# Dataset

We use CICIDS 2017 to test for transferability in learning between attack classes. The dataset can be downloaded at https://www.unb.ca/cic/datasets/ids-2017.html


# Training
To start centralised machine learning and transferability analysis, run train.py. This code trains the model with one class of data and tests on all other classes.

Arguments:

--bootstrap True/False(Default) to enable bootstrapping during training

python train.py


# Results

The results of our work can be found in results/ in normal testing case, differential inputs and temporal averaging. The pickle files have the confusion matrices stored for normal testing case and transformed testing cases. The weights for each training are also available.
