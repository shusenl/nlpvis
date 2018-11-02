/*
Display sentences and apply perturbation to the sentences
*/

class sentenceComponent extends baseComponent {
    constructor(uuid) {
        super(uuid);
        this.subscribeDatabyNames(["sentenceList", "currentPair"]);

        //init
        this.callFunc("initSetup");

        //setup UI
        d3.select(this.div + "perturbTarget").on("click", this.perturbTarget
            .bind(this));
        d3.select(this.div + "perturbSource").on("click", this.perturbSource
            .bind(this));

        d3.select(this.div + "Predict").on("click", d => {
            this.callFunc("predict");
        });
        d3.select(this.div + "PredictAll").on("click", d => {
            //produce all combinations
            this.callFunc("predictAll");
        });

        //update data when currentPair changes
        d3.select(this.div + "src").on("change", this.onUpdateCurrentPair.bind(
            this));
        d3.select(this.div + "targ").on("change", this.onUpdateCurrentPair.bind(
            this));
    }

    draw() {
        //add to sentence list
        // if (this.data["sentenceList"] !== undefined) {
        //     this.onReceiveSentenceList();
        // }

        //update currentPair display
        if (this.data["currentPair"]["sentences"] !== undefined) {
            this.onReceiveCurrentPair();
        }
    }

    parseDataUpdate(msg) {
        super.parseDataUpdate(msg);
        // console.log(msg);

        switch (msg['name']) {
            case "sentenceList":
                this.onReceiveSentenceList();
                break;
            case "currentPair":
                let pair = msg["data"]["data"]["sentences"];
                // console.log("sentenceComponent:", pair);
                if (this.oldPair) {
                    if (this.oldPair[0].split(" ").length !== pair[0].split(
                            " ").length ||
                        this.oldPair[1].split(" ").length !== pair[1].split(
                            " ").length
                    ) {
                        this.clearDropdown(this.div + "srcInput");
                        this.clearDropdown(this.div + "targInput");
                    }
                    //reset perturbed sentences
                    this.data["allSourceSens"] = undefined;
                    this.data["allTargetSens"] = undefined;
                }

                this.onReceiveCurrentPair();
                this.oldPair = JSON.parse(JSON.stringify(pair));
                break;
        }
    }

    parseFunctionReturn(msg) {
        super.parseFunctionReturn(msg);
        switch (msg['func']) {
            case 'perturbSentence':
                this.updatePerturbedSentences(msg["data"]["data"], msg[
                    "data"]);
                break;
        }
    }


    /////////////////// event handler /////////////////////
    onReceiveSentenceList() {
        // console.log("sentenceList:", this.data["sentenceList"]);
        d3.select(this.div + "selectExample")
            .on("change", this.onChangeOriginalPair.bind(this));
        var options = d3.select(this.div + "selectExample")
            .selectAll('option')
            .data(this.data["sentenceList"]).enter()
            .append('option')
            .text(function(d) {
                return d["src"].substring(4) + " | " + d["targ"].substring(
                    4);
            })
            .property("value", (d, i) => i);
    }

    onReceiveCurrentPair() {
        var currentPair = this.data['currentPair']["sentences"];

        this.source = currentPair[0].substring(4);
        this.target = currentPair[1].substring(4);

        d3.select(this.div + "src").property("value", this.source);
        d3.select(this.div + "targ").property("value", this.target);

        // console.log("----------", this.data["allSourceSens"]);
        if (this.data["allSourceSens"]) {
            $(this.div + "src").highlightWithinTextarea({
                highlight: this.getSentenceDiff(
                    this.data["allSourceSens"][0].substring(4),
                    currentPair[0].substring(4)), //
                className: 'blue'
            });
        }
        if (this.data["allTargetSens"]) {
            $(this.div + "targ").highlightWithinTextarea({
                highlight: this.getSentenceDiff(
                    this.data["allTargetSens"][0].substring(4),
                    currentPair[1].substring(4)), //
                className: 'blue'
            });
        }
        // console.log(currentPair);
    }

    onChangeOriginalPair() {
        var index = Number(d3.select(this.div + "selectExample").property(
            'value'));
        // console.log(index);
        var currentPair = [this.data["sentenceList"][index]["src"],
            this.data["sentenceList"][index]["targ"]
        ];
        // var groundTruthLabel = this.data["sentenceList"][index]["pred"]
        // console.log(groundTruthLabel);
        // this.onReceiveCurrentPair()
        this.data["currentPair"] = {
            "sentences": currentPair,
            "label": this.data["sentenceList"][index]["pred"]
        };
        d3.select(this.div + "src").property("value", currentPair[0].substring(
            4));
        d3.select(this.div + "targ").property("value", currentPair[1].substring(
            4));

        //update rest of the views
        this.setData("currentPair", this.data["currentPair"]);
        // this.setData("groundTruthLabel", groundTruthLabel);

        this.clearPreviousPair();
    }

    clearPreviousPair() {
        this.clearDropdown(this.div + "srcInput");
        this.clearDropdown(this.div + "targInput");

        //reset allSens
        let currentPair = this.data["currentPair"]["sentences"];
        this.setData("allSourceSens", [currentPair[0]]);
        this.setData("allTargetSens", [currentPair[1]]);
    }

    onUpdateCurrentPair() {
        var currentPair = ["<s> " + d3.select(this.div + "src").property(
                "value"),
            "<s> " + d3.select(this.div + "targ").property("value")
        ];
        this.data["currentPair"]["sentences"] = currentPair;
        this.setData("currentPair", this.data["currentPair"]);
    }

    updatePerturbedSentences(sentences) {
        //sentences[0] is the unperturbed sentence
        if (this.data["currentPair"]["sentences"][0] === "<s> " + sentences[
                0]) {
            this.setData("allSourceSens", sentences.map(d => "<s> " + d));
            this.addDropdown(this.div + "srcInput", sentences, this.div +
                "src");
        } else if (this.data["currentPair"]["sentences"][1] === "<s> " +
            sentences[0]) {
            this.setData("allTargetSens", sentences.map(d => "<s> " + d));
            this.addDropdown(this.div + "targInput", sentences, this.div +
                "targ");
        }
    }

    perturbSource() {
        if (this.data["currentPair"]["sentences"] !== undefined) {
            this.callFunc("perturbSentence", {
                "sentence": this.data["currentPair"]["sentences"][0]
                    .substring(4)
            });
        }
    }

    perturbTarget() {
        if (this.data["currentPair"]["sentences"] !== undefined) {
            this.callFunc("perturbSentence", {
                "sentence": this.data["currentPair"]["sentences"][1]
                    .substring(4)
            });
        }
    }

    clearDropdown(selector) {
        d3.select(selector).select(".dropdown-toggle").remove();
        d3.select(selector).select(".dropdown-menu")
            .selectAll(".dropdown-item").remove();
    }

    addDropdown(selector, sentences, labelSelector) {
        if (d3.select(selector).select(".dropdown-toggle").empty()) {
            // console.log("add Button");
            d3.select(selector).append("button")
                .attr("class",
                    "btn btn-outline-secondary dropdown-toggle dropdown-toggle-split"
                )
                .attr("data-toggle", "dropdown")
                // .attr("aria-haspopup", "true")
                // .attr("aria-expanded", "false")
                .append("span")
                .attr("class", "sr-only");

            d3.select(selector)
                .append("div")
                .attr("class", "dropdown-menu");

            //cleanup
            // console.log(menu);
            var menu = d3.select(selector).select(".dropdown-menu");

            menu.selectAll(".dropdown-item").remove();
            menu.selectAll(".dropdown-item")
                .data(sentences)
                .enter()
                .append("a")
                .attr("class", "dropdown-item")
                .html(d => this.colorSentenceDiff(sentences[0], d))
                .on("click", (d, i) => {
                    //update sentence edit box
                    d3.select(labelSelector).property("value", d);
                    $(labelSelector).highlightWithinTextarea({
                        highlight: this.getSentenceDiff(
                            sentences[0],
                            d), //
                        className: 'blue'
                    });

                    this.onUpdateCurrentPair();
                });
        }
        /////////////////// reference /////////////////
        // <button type="button" class="btn btn-outline-secondary dropdown-toggle dropdown-toggle-split" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
        //   <span class="sr-only">Toggle Dropdown</span>
        // </button>
        // <div class="dropdown-menu" id="{{id}}selectTarget">
        //   <a class="dropdown-item" href="#">Action</a>
        // </div>
    }

    ////////////////////// helper /////////////////////

    colorSentenceDiff(origin, perturbed) {
        var originList = origin.split(" ");
        var perturbedList = perturbed.split(" ");
        let originWords = new Set(originList);
        // if (originList.length === perturbedList.length) {
        var outputStr = "";
        for (var i = 0; i < perturbedList.length; i++) {
            var word = perturbedList[i];
            // if (word !== originList[i] && word !== ".")
            if (!originWords.has(word)) {
                // console.log(word, "-", originList[i]);
                word = "<span style=\"background:#87CEFA\">" + word +
                    "</span>";
            }
            word += " "

            outputStr += word;
        }
        // <span style="color:#FF0000">some text</span>
        return outputStr;
    }

    getSentenceDiff(origin, perturbed) {
        let originList = origin.split(" ");
        let perturbedList = perturbed.split(" ");
        let wordList = [];
        let originWords = new Set(originList);
        // console.log(originWords);

        // if (originList.length === perturbedList.length) {
        //     var outputStr = "";
        //     for (var i = 0; i < originList.length; i++) {
        //         var word = perturbedList[i];
        //         if (word !== originList[i] && word !== ".") {
        //             // console.log(word, "-", originList[i]);
        //             wordList.push(word);
        //         }
        //     }
        // }

        //check which word is not appeared in the original
        for (var i = 0; i < perturbedList.length; i++) {
            var word = perturbedList[i];
            if (!originWords.has(word)) {
                wordList.push(word);
            }
        }

        // console.log(wordList);
        return wordList;
    }
}
