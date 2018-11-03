
'''
 - the portal for sending data to the visualization
 - act as a server for the js based visualization
 - open web browser


 Data Model
 - data object is linked by object id, json format
'''

from flask import Flask
import socketio
import eventlet
import webbrowser, threading
from .NLPutility import tokenizer
from .socketioManager import *

import time
import json

app = Flask(__name__)
sio = socketio.Server()
fApp = socketio.Middleware(sio, app)
dataManager = socketioManager(sio)

pipelineState = [{
    "index":0,
    "name": "encoder",
    "histName": "Gradient Histo",
    "state": True,
    "arrow": [1]
    }, {
    "index":1,
    "name": "attention",
    "histName": "Gradient Histo",
    "state": True,
    "arrow": [2]
    }, {
    "index":2,
    "name": "classifier",
    "histName": "Gradient Histo",
    "state": True,
    "arrow": []
    }
];

#################### server control ######################
layoutConfig = None

class visModule(object):
    def __init__(self, componentLayout):
        global layoutConfig
        layoutConfig = componentLayout
        dataManager.setObject(self)
        self.parserHook = None
        self.sentencePerturbationHook = None

    # envoke callback when the server is running
    @sio.on('message', namespace='/app')
    def parsingMessage(sid, msg):
        # print sid, msg
        return dataManager.receiveFromClient(msg)

    @app.route('/<name>')
    def views(name):
        return app.send_static_file('viewTemplates/'+name+".mst")

    @app.route('/')
    def index():
        dataManager.clear()
        dataManager.setData("componentLayout", layoutConfig)
        return app.send_static_file('index.html')

    ##### placeholder to be implemented in the derived class
    def initSetup(self):
        pass

    def show(self):
        url = 'http://localhost:5050'
        threading.Timer(1.5, lambda: webbrowser.open(url, new=0) ).start()
        eventlet.wsgi.server(eventlet.listen(('localhost', 5050)), fApp)

        # deploy as an eventlet WSGI server
        # sio.start_background_task(self.startServer)

    # @staticmethod
    def startServer(self, port=5050):
        eventlet.wsgi.server(eventlet.listen(('localhost', port)), fApp)
        # socketio.run(app, host='localhost',port=5050, debug=True)

    def setSentenceParserHook(self, callback):
        self.parserHook = callback

    def setSentencePerturbationHook(self, callback):
        self.sentencePerturbationHook = callback

    def perturbSentence(self, sentence):
        if self.sentencePerturbationHook:
            perturbed = self.sentencePerturbationHook(sentence)
            if str(perturbed[0])!=str(sentence):
                return [sentence] + perturbed
            else:
                return perturbed
        else:
            print ("No sentence perturbator is specified!")
            return False

    #get sentence parse tree
    def parseSentence(self, sentence):
        if self.parserHook:
            depTree = self.parserHook(sentence)
            return {"depTree": depTree, "sentence":sentence}
        else:
            print ("No sentence parser is specified!")
            return False
