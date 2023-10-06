# Intrusion-Detection-on-NSL-KDD

A practical implementation of an intrusion detection system described in the following paper:《An Intrusion Detection System Using a Deep Neural Network with Gated Recurrent Units》（DOI：10.1109/ACCESS.2018.DOI）

**Note that I am not the original author of the paper!**

This project is based on ***Keras*** API

### Docker image configuration (optional)

Keras Docker image：

https://hub.docker.com/r/gw000/keras

tag：:2.1.4-py3-tf-gpu

docker：keras-py3-tf-gpu:2.1.4

CPU：

`$ docker run -it --rm -v $(pwd):/srv gw000/keras:2.1.4-py3-tf-gpu /srv/run.py`

GPU:

`$ docker run -it --rm $(ls /dev/nvidia* | xargs -I{} echo '--device={}') $(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') -v $(pwd):/srv gw000/keras:2.1.4-py3-tf-gpu /srv/run.py`

### Dataset

NSL_KDD dataset:

https://www.unb.ca/cic/datasets/nsl.html

More information about NSL_KDD dataset：

https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657

### Installation

Use the following command to install dependencies:

`pip install -r requirements.py`

### Usage

`python3 run.py`

### Results

By using 20 Epochs for training the model the Accuracy of 98%+ could be achieved, although it lowers to about 96% after using Dropout.
