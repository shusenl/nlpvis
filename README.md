# NLPVis

## Intro
- This software is designed to visualize the attention of neural network based natural language models. Beside the visualization code, we also includes some state-of-the-art natural language inference (NLI) and machine  comprehension (MC) model.  
The NLI learning code was extensively modified from Harvard NLP's reimplimentation of Ankur Parikh's decomposable attention model https://github.com/harvardnlp/decomp-attn.
The MC model is based on the BiDAF implementation.

## Setup

### 1. Install
- Please install numpy, pytorch, h5py, requests, nltk, python-socketio, eventlet, pattern, etc
   `pip install -r requirements.txt`
- Download model and data file (download from google drive):  
   `cd src; python downloadModels.py`

### 1. Training
- The pre-trained model will be loaded.

### 2. Test the model
- Using the pretrained model to do evaluation on val set. Expect to see `Val: 0.8631, Loss: 0.3750`
- To test run the following:  
  `python eval.py --gpuid -1 --data ../data/snli_1.0/snli_1.0-val.hdf5 --word_vecs ../data/glove.hdf5 --encoder proj --attention local --classifier local --dropout 0.0 --load_file ../data/local_300_parikh`



### 3. Run the visualization server for NLI (for MC run MCexampleVis.py)
 - Start the server:  
   `python NLIexampleVis.py`
 - Then open the browser at http://localhost:5050/
