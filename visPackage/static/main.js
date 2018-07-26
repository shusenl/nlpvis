/////////////////////////// global states ////////////////////////////
var namespace = '/app'; //global namespace
//create a web socket connect to the server domain.
var socket = io('http://' + document.domain + ':' + location.port + namespace);
// var socket = io.connect('http://' + document.domain + ':' + location.port + namespace);

var panelMetaInfo = {
    'Prediction': ['prediction_view', 'predictionComponent'],
    'AttentionMatrix': ['template_view', 'attentionMatrixComponent'],
    'Sentence': ['sentence_view', 'sentenceComponent'],
    "AttentionGraph": [
        'template_view',
        'attentionGraphComponent'
    ],
    "AttentionAsymmetric": ['asymmetric_view',
        "attentionAsymmetricComponent"
    ],
    "AttentionSubMatrix": ["template_view", "attentionSubMatrixComponent"],
    'Summary': ['evaluation_view', 'evaluationComponent'],
    'Pipeline': ['template_view', 'pipelineComponent'],
    'LatentRepresentation': ['latent_view', 'latentSpaceComponent'],
    "Paragraph": ['paragraph_view', 'paragraphComponenet']
};

//for lookup component class on-the-fly
var objectMap = {
    predictionComponent: predictionComponent,
    attentionMatrixComponent: attentionMatrixComponent,
    attentionGraphComponent: attentionGraphComponent,
    sentenceComponent: sentenceComponent,
    evaluationComponent: evaluationComponent,
    pipelineComponent: pipelineComponent,
    latentSpaceComponent: latentSpaceComponent,

    paragraphComponenet: paragraphComponenet,
    attentionAsymmetricComponent: attentionAsymmetricComponent,
    attentionSubMatrixComponent: attentionSubMatrixComponent
};

/////////////////////////// create layout ///////////////////////////
// var appLayout = new window.GoldenLayout(config, $('body')); //
var visLayout = new glayout($('body'), panelMetaInfo, objectMap);

// appLayout.init()
// handle whole window resize
window.addEventListener('resize', function(size) {
    // console.log(size);
    // appLayout.updateSize();
    visLayout.updateSize();
})

window.onbeforeunload = function(e) {
    console.log("@@@@@@@@@@@ reset module on server @@@@@@@@@\n");
    $.get("/", d => console.log(d));
    // $.get("/reset/", d => console.log(d));
};
