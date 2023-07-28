# HINTBOT

This repo provides a reference implementation of **HINTBOT** as described in the paper:
> HintBot: An Intelligent End-to-end Pipeline for Prediction of Large Scale P2P Botnet Infection
> 
> Submitted for publication.

## Basic Usage

### Requirements

The code was tested with Python 3.6, `tensorflow-gpu` 1.7 and Cuda 9.0. Install the dependencies:

```python
pip install -r requirements.txt
```

### How to run the code
```shell
>  python gen_walks.py
>  python preprocess.py
>  python run.py
```

#### Options
You may change the model settings manually in `model.py` or directly into the codes. 

#### Datasets
See some data in ./datasets/README.md
