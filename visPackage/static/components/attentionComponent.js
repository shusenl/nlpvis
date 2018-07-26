/*
Based class for attention visualization
*/

//beside margin matrix will take 2/3 width and 2/3 height space
class attentionComponent extends baseComponent {
    constructor(uuid) {
        super(uuid);
        this.subscribeDatabyNames(["attention", "normAttentionCol",
            "normAttentionRow", "currentPair", "highlight",
            "attentionDirection"
        ]);

        this.margin = {
            top: 10,
            right: 10,
            bottom: 10,
            left: 10
        };

        //init member
        this.srcIndexMaskSet = new Set();
        this.targIndexMaskSet = new Set();
        this.aggregationIndex = {};
        this.screenshotIcon = true;
    }

    clear() {
        // console.log("clear: based class");
        if (this.svgContainer)
            d3.select(this.svgContainer).remove();
    }

    initSvg() {
        //create svg
        if (this.svgContainer === undefined) {
            this.svgContainer = d3.select(this.div).append("svg")
                .attr("width", this.pwidth)
                .attr("height", this.pheight);
            this.svg = this.svgContainer
                .append("g")
                .attr("transform", "translate(" + this.margin.left + "," +
                    this.margin.top + ")");

            if (this.screenshotIcon) {
                this.svgSave = new svgExporter(this.svgContainer, [this.width -
                    10, 10
                ]);
            }
        } else {
            this.svgContainer
                .attr("width", this.pwidth)
                .attr("height", this.pheight)

            this.svg.selectAll(
                "text,rect,path, #attData, defs").remove();

            if (this.screenshotIcon) {
                this.svgSave.updatePos([this.width - 10, 10]);
                this.svgSave.draw();
            }
        }
    }

    sen2words(sen) {
        var words = sen.match(/\S+/g);
        // words.unshift("\<s\>");
        return words;
    }

    collapSenBySet(words, maskSet) {
        var collapWords = [];
        for (var i = 0; i < words.length; i++) {
            if (!maskSet.has(i))
                collapWords.push(words[i]);
        }
        return collapWords;
    }

    collapseSrc(mask) {
        this.srcIndexMaskSet.clear();
        mask.map((d, i) => {
            if (d === 0) {
                this.srcIndexMaskSet.add(i)
            }
        });
    }

    collapseTarget(mask) {
        this.targIndexMaskSet.clear();
        mask.map((d, i) => {
            if (d === 0) {
                this.targIndexMaskSet.add(i);
            }
        });
    }

    resize() {
        //you can redraw or resize your vis here
        this.draw();
    }

    attUpdate() {
        // console.log(this.normAttention);
        this.setData("normAttentionCol", this.normAttentionCol);
        this.setData("normAttentionRow", this.normAttentionRow);
        this.callFunc("attentionUpdate", {
            "att_soft1": this.normAttentionRow,
            "att_soft2": this.normAttentionCol
        });
    }

    //convert raw attention to normalized attention
    //orientation specifiy whether we can normalize row / col
    convertRawAtt(raw, orientation = "row") {
        let normAttention;
        let transpose = m => m[0].map((x, i) => m.map(x => x[i]));
        if (orientation === "row") {
            normAttention = raw.map(d => this.softmax(d));
        } else if (orientation === "col") {
            // normAttention = raw.map(d => this.softmax(d));
            let raw_T = transpose(raw);
            let norm_T = raw_T.map(d => this.softmax(d));
            normAttention = transpose(norm_T);
        } else {
            //if no option is specifiy return raw
            normAttention = raw;
        }
        return normAttention;
    }

    swapAttDirection() {
        if (this.attentionDirection === 'row') {
            // this.normAttention = this.convertRawAtt(this.rawAttention,
            // 'col');
            this.normAttention = this.normAttentionCol;
            this.attentionDirection = 'col';
            this.setData("attentionDirection", 'col');
            this.draw();
        } else if (this.attentionDirection === 'col') {
            // this.normAttention = this.convertRawAtt(this.rawAttention,
            // 'row');
            this.normAttention = this.normAttentionRow;
            this.attentionDirection = 'row';
            this.setData("attentionDirection", 'row');
            this.draw();
        }
    }

    parseDataUpdate(msg) {
        super.parseDataUpdate(msg);
        switch (msg["name"]) {
            case "attention":
                // console.log(this.data["attention"]);
                //if attention is updated, redraw attention
                // this.srcDepTreeData = undefined;
                // this.targDepTreeData = undefined;
                //normalize att
                if (this.rawAttention) {
                    //clone the raw attention
                    this.preRawAtt = JSON.parse(JSON.stringify(this.rawAttention));
                }
                this.rawAttention = this.data["attention"];
                this.attentionDirection = 'row';
                this.normAttentionRow = this.convertRawAtt(this.rawAttention,
                    'row');

                this.normAttentionCol = this.convertRawAtt(this.rawAttention,
                    'col');

                this.normAttention = this.normAttentionRow;
                // console.log(this.rawAttention);
                // console.log(this.normAttention);

                this.draw();

                //parse the sentence
                let currentPair = this.data["currentPair"]["sentences"];
                if (this.srcDepTreeData === undefined) {
                    this.callFunc("parseSentence", {
                        "sentence": currentPair[0]
                    });
                }
                if (this.targDepTreeData === undefined) {
                    this.callFunc("parseSentence", {
                        "sentence": currentPair[1]
                    });
                }

                break;

            case "normAttentionCol":
                this.normAttentionCol = this.data["normAttentionCol"];
                if (this.attentionDirection === 'col') {
                    // console.log("update normAttentionCol");
                    this.normAttention = this.normAttentionCol;
                    this.draw();
                }
                break;

            case "normAttentionRow":
                this.normAttentionRow = this.data["normAttentionRow"];
                if (this.attentionDirection === 'row') {
                    // console.log("update normAttentionRow");
                    this.normAttention = this.normAttentionRow;
                    this.draw();
                }
                break;

            case "currentPair":
                let pair = msg["data"]["data"][
                    "sentences"
                ];
                // console.log("pair", pair);

                if (this.oldPair) {
                    //clear the current dependency
                    if (this.oldPair[0].split(" ").length !== pair[0].split(
                            " ").length ||
                        this.oldPair[1].split(" ").length !== pair[1].split(
                            " ").length
                    ) {
                        console.log("new pair loaded, clear tree/att");
                        if (this.svg)
                            this.svg.selectAll("*").remove();
                        this.srcDepTreeData = undefined;
                        this.src_dep = undefined;
                        this.targDepTreeData = undefined;
                        this.targ_dep = undefined;
                        this.normAttention = undefined;
                        this.setData("allSourceSens", [pair[0]]);
                        this.setData("allTargetSens", [pair[1]]);
                    }
                } else {

                }

                this.srcWords = pair[0].match(/\S+/g);
                this.targWords = pair[1].match(/\S+/g);
                this.oldPair = JSON.parse(JSON.stringify(pair));
                break;

            case "attentionDirection":
                // console.log("attentionDirection is changed\n");
                let direction = msg["data"]["data"];
                if (this.attentionDirection !== direction) {
                    this.swapAttDirection();
                }
                break;

            case "highlight":
                let srcIndex = msg["data"]["data"][0];
                let targIndex = msg["data"]["data"][1];
                this.handleHighlightEvent(srcIndex, targIndex);
                break;
        }
    }

    parseFunctionReturn(msg) {
        super.parseFunctionReturn(msg);
        switch (msg["func"]) {
            case "parseSentence":
                this.handleParsedSentence(msg["data"]["data"]);
        }
    }

    handleParsedSentence(parseResult) {
        let parsedSen = parseResult["sentence"];
        if (parsedSen == this.data["currentPair"]["sentences"][0]) {
            //draw structure
            this.srcDepTreeData = parseResult["depTree"];
            this.drawDepTree();
        }

        if (parsedSen == this.data["currentPair"]["sentences"][1]) {
            this.targDepTreeData = parseResult["depTree"];
            this.drawDepTree();
        }
    }

    aggregationMatrix(root, nodes) {
        //check whether the aggregate root already exist
        if (root in this.aggregationIndex) {
            delete this.aggregationIndex[root];
        } else {
            this.aggregationIndex[root] = nodes;
        }

        //clone object
        this.aggregatedMatrix = this.normAttention.map(function(arr) {
            return arr.slice();
        })

        //TODO: aggregate the information base on this.normAttention
        for (const root in this.aggregationIndex) {
            this.aggregationMatrixHelper(root, this.aggregationIndex[
                root]);
        }

    }

    aggregationMatrixHelper(root, indexs) {

        for (let i = 0; i < this.aggregatedMatrix.length; i++) {
            let maxvalue = this.aggregatedMatrix[i][root];
            for (let j = 0; j < this.aggregatedMatrix[i].length; j++) {
                if (indexs.includes(j)) {
                    if (maxvalue < this.aggregatedMatrix[i][j]) {
                        maxvalue = this.aggregatedMatrix[i][j];
                    }
                    this.aggregatedMatrix[i][j] = 0;
                }
            }
            this.aggregatedMatrix[i][root] = maxvalue;
        }
    }

    softmax(arr) {
        return arr.map(function(value, index) {
            return Math.exp(value) / arr.map(function(y /*value*/ ) {
                return Math.exp(y)
            }).reduce(function(a, b) {
                return a + b
            })
        })
    }

    normalization(arr) {
        return arr.map(function(value, index) {
            return value / arr.map(function(y /*value*/ ) {
                return y;
            }).reduce(function(a, b) {
                return a + b;
            })
        });
    }

    //normalize col of the input matrix
    normalizeCol(mat, col) {
        var sum = mat.map(d => d[col]).reduce((a, b) => a + b, 0.0);
        // console.log(sum);
        for (var i = 0; i < mat.length; i++)
            mat[i][col] = mat[i][col] / sum;
        return mat;
    }

    toggleAttMode(mode) {
        if (mode === "C") { //current
            this.normAttention = this.convertRawAtt(this.rawAttention);
            this.draw();
        } else if (mode === "P") { //previous
            if (this.preRawAtt) {
                this.normAttention = this.convertRawAtt(this.preRawAtt);
                this.draw();
            }
        } else if (mode === "D") { //difference
            if (this.preRawAtt) {
                this.normAttention = this.attDiff(this.convertRawAtt(this.rawAttention),
                    this.convertRawAtt(this.preRawAtt));
                this.comparisonFlag = true;
                this.draw();
                this.comparisonFlag = false;
            }
        }
    }

    //assume mat1, mat2 have the same dimension
    attDiff(mat1, mat2) {
        var mat = JSON.parse(JSON.stringify(mat1));
        for (let i = 0; i < mat1.length; i++)
            for (let j = 0; j < mat1[i].length; j++) {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            }
        return mat;
    }

    drawAttDataToggle(svg, pos, orientation = "vertical") {

        // swap attention normalization button
        if (this.svg.select("#attData").empty()) {
            this.swapButton = this.svg.append("g").attr("id", "attData");
            this.swapButton.append("rect")
                .attr("rx", 3)
                .attr("ry", 3)
                .attr("x", pos[0])
                .attr("y", pos[1])
                .attr("width", d => {
                    if (orientation === 'vertical') {
                        return 20;
                    } else {
                        return 40;
                    }
                })
                .attr("height", d => {
                    if (orientation === 'vertical') {
                        return 30;
                    } else {
                        return 20;
                    }
                })
                .style("stroke", "lightgrey")
                .style("fill", "white")
                .on("click", d => {
                    this.swapAttDirection();
                })
                .on("mouseover", function(d) {
                    d3.select(this).style("fill", "lightgrey");
                }).on("mouseout", function(d) {
                    d3.select(this).style("fill", "white");
                });
            this.swapButton.append("text")
                .attr("x", d => {
                    if (orientation === 'vertical') {
                        return pos[0] + 10;
                    } else {
                        return pos[0] + 5;
                    }
                })
                .attr("y", d => {
                    if (orientation === 'vertical') {
                        return pos[1] + 1;
                    } else {
                        return pos[1] + 10;
                    }
                })
                .text(d => {
                    if (this.attentionDirection === "col")
                        return "att2";
                    else if (this.attentionDirection === "row")
                        return "att1";
                })
                .style("text-anchor", "start")
                .style("fill", "grey")
                .style("writing-mode", d => {
                    if (orientation === 'vertical') {
                        return "vertical-rl";
                    } else {
                        return "hortizontal-rl";
                    }
                })
                .style("alignment-baseline", "middle")
                .style("pointer-events", "none");
        } else {
            console.log("update swap button!!");
            this.svg.select("#swapButton")
                .select("rect")
                .attr("x", pos[0])
                .attr("y", pos[1]);

            this.svg.select("#swapButton")
                .select("text")
                .attr("x", d => {
                    if (orientation === 'vertical') {
                        return pos[0] + 10;
                    } else {
                        return pos[0] + 20;
                    }
                })
                .attr("y", d => {
                    if (orientation === 'vertical') {
                        return pos[1] + 20;
                    } else {
                        return pos[1] + 10;
                    }
                });
        }

    }

    drawAttDisplayToggle(svg, pos) {
        let label = ["C", "P", "D"]
        let toggle = svg.append("g");
        toggle.selectAll(".attToggle")
            .data(label)
            .enter()
            .append("rect")
            .attr("class", "attToggle")
            .attr("rx", 3)
            .attr("ry", 3)
            .attr("x", (d, i) => pos[0] + i * 22)
            .attr("y", pos[1])
            .attr("width", 20)
            .attr("height", 20)
            .attr("fill", "lightgrey")
            .on("click", d => {
                this.toggleAttMode(d);
            }).on("mouseover", function(d) {
                d3.select(this).attr("fill", "grey");
            }).on("mouseout", function(d) {
                d3.select(this).attr("fill", "lightgrey");
            });

        toggle.selectAll(".toggleLabel")
            .data(label)
            .enter()
            .append("text")
            .attr("class", "toggleLabel")
            .attr("x", (d, i) => pos[0] + 10 + i * 22)
            .attr("y", pos[1] + 10)
            .text(d => d)
            .style("text-anchor", "middle")
            .style("alignment-baseline", "middle")
            .style("pointer-events", "none");
    }
}
