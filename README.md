# NLPVis

## Intro
- This software is designed to visualize the attention of neural network based natural language models. Beside the visualization code, we also includes some state-of-the-art natural language inference (NLI) and machine  comprehension (MC) model.  
The NLI learning code was extensively modified from Harvard NLP's reimplimentation of Ankur Parikh's decomposable attention model https://github.com/harvardnlp/decomp-attn.
The MC model is based on the BiDAF implementation.

[![Video Demonstration](https://img.youtube.com/vi/PKiM4i0oIuY/0.jpg)](https://www.youtube.com/watch?v=PKiM4i0oIuY)

## Setup

### 1. Install
- The tool has only been tested with Python 2.7, support/test for Python 3.x is planned.
- Please install numpy, pytorch, h5py, requests, nltk, python-socketio, eventlet, pattern, etc  
   `pip install -r requirements.txt`
- Download model and data file (download from google drive):  
   `python downloadModels.py`

### 1. Training
- The pre-trained model will be loaded.

### 2. Test the model
- Using the pretrained model to do evaluation on val set. Expect to see `Val: 0.8631, Loss: 0.3750`
- To test run the following:  
  `python -m nli_src.eval --gpuid -1 --data data/snli_1.0/snli_1.0-val.hdf5 --word_vecs data/glove.hdf5 --encoder proj --attention local --classifier local --dropout 0.0 --load_file data/local_300_parikh`


### 3. Run the visualization server for NLI (for MC run MCexampleVis.py)
 - Start the server:  
   `python NLIexampleVis.py`
 - Then open the browser at http://localhost:5050/

### 4. Docker Image
Alternatively you can run the server from a docker image without installing anything.
https://hub.docker.com/r/dockerzhimin/nlpvis/

Instruction to run the docker server:
- Download docker image:  
  `docker pull dockerzhimin/nlpvis`
- look up docker image name: 
  `docker ls`
- Start the server (replace image_name with the correct image name):  
  `docker run -d -p 5050:5050 [image_name]`
- Then open the browser at http://localhost:5050/
  The tool has only been extensively tested for Chrome, there are known issues with firefox (font size and positions are altered for some visual elements)

Related Publication:

**Visual Interrogation of Attention-Based Models for Natural Language Inference and Machine Comprehension** 

Shusen Liu, Tao, Li, Zhimin Li, Vivek Srikumar, Valerio Pascucci, and Peer-Timo Bremer .
Conference on Empirical Methods in Natural Language Processing (EMNLP) Demonstration Track, 2018.
