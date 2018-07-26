/*
the triangle vis of the prediction result
*/

class predictionComponent extends baseComponent {
    constructor(uuid) {
        super(uuid);

        //dict for prediction (groundTruth, p[3])
        //subscribe to data
        this.subscribeDatabyNames([
            "allSourceSens", "allTargetSens",
            "prediction", "allPairsPrediction",
            "predictionUpdate", "currentPair", "pipeline",
            "predictionBatchUpdate",
            "learningRate",
            "C_mira",
            "iteration"
        ]);

        this.margin = {
            top: 25,
            right: 25,
            bottom: 25,
            left: 25
        };

        this.svgContainer = d3.select(this.div);
        this.svgSave = new svgExporter(this.svgContainer);

        this.draw();
    }

    draw() {
        // console.log(this.data);
        this._updateWidthHeight();
        this.svg = d3.select(this.div + "triangle");

        this.svgSave.updatePos([this.width, 10])
        this.svgSave.draw();

        //draw tooltip
        if (this.legend === undefined) {
            this.legend = this.svg.append("g");
            this.legend.append("circle")
                .attr("class", "legend")
                .attr("cx", 0)
                .attr("cy", 0)
                .attr("r", 6)
                .attr("fill", "gold")
                .style("stroke", "white");
            this.legend.append("text")
                .attr("x", 10)
                .attr("y", 0)
                .text("Original")
                .style("alignment-baseline", "middle")
                .style("pointer-events", "none")
                .style("font-size", 10);
            this.legend.append("circle")
                .attr("class", "legend")
                .attr("cx", 0)
                .attr("cy", 15)
                .attr("r", 3)
                .attr("fill", "grey")
                .style("stroke", "white");
            this.legend.append("text")
                .attr("x", 10)
                .attr("y", 15)
                .text("Perturbed")
                .style("alignment-baseline", "middle")
                .style("pointer-events", "none")
                .style("font-size", 10);
            this.legend.append("rect")
                .attr("class", "legend")
                .attr("x", -3)
                .attr("y", 30 - 3)
                .attr("width", 6)
                .attr("height", 6)
                .attr("fill", "lightblue")
                .style("stroke", "white");
            this.legend.append("text")
                .attr("x", 10)
                .attr("y", 30)
                .text("Optimized")
                .style("alignment-baseline", "middle")
                .style("pointer-events", "none")
                .style("font-size", 10);

            //ground truth
            this.legend.append("rect")
                .attr("rx", 3)
                .attr("ry", 3)
                .attr("x", -10)
                .attr("y", 40)
                .attr("width", 20)
                .attr("height", 15)
                .attr("stroke", "none")
                .attr("fill", "lightgreen");
            this.legend.append("text")
                .attr("x", 12)
                .attr("y", 47)
                .text("Ground Truth")
                .style("alignment-baseline", "middle")
                .style("pointer-events", "none")
                .style("font-size", 10);

            //prediction update
        }

        //entailment
        //neutral, Contradiction, Entailment
        //112,0 0,194 224,194

        this.svg.select(this.div + "label").remove();
        var label = this.svg.append("g").attr("id", this.uuid + "label");
        this.Neutral = label.append('rect')
            .attr("rx", 3)
            .attr("ry", 3)
            .attr("x", 112 - 23)
            .attr("y", -20)
            .attr("width", 50)
            .attr("height", 20)
            .attr("stroke", "none")
            .attr("fill", "none");

        label.append('text')
            // .attr("class", "trilabel")
            .attr("x", 112 - 20)
            .attr("y", -7)
            .text("Neutral")
            .style("font-size", "14px")
            .style("fill", "grey");

        //neutral
        this.Contradiction = label.append('rect')
            .attr("rx", 3)
            .attr("ry", 3)
            .attr("x", 0 - 10)
            .attr("y", 194 + 4)
            .attr("width", 80)
            .attr("height", 20)
            .attr("stroke", "none")
            .attr("fill", "none");

        label.append('text')
            // .attr("class", "trilabel")
            .attr("x", 0 - 10)
            .attr("y", 194 + 17)
            .text("Contradiction")
            .style("font-size", "14px")
            .style("fill", "grey");

        //contradiction
        this.Entailment = label.append('rect')
            .attr("rx", 3)
            .attr("ry", 3)
            .attr("x", 224 - 55)
            .attr("y", 194 + 4)
            .attr("width", 63)
            .attr("height", 20)
            .attr("stroke", "none")
            .attr("fill", "none");

        label.append('text')
            // .attr("class", "trilabel")
            .attr("x", 224 - 55)
            .attr("y", 194 + 17)
            .text("Entailment")
            .style("font-size", "14px")
            .style("fill", "grey");
        // updateData(sdata, 0);

        this.svg.attr("width", this.pwidth)
            .attr("height", this.pheight)
            .attr("x", 0)
            .attr("y", 0);

        // this.updateSelection();
        //entail, netrual, contradict
        // var p1 = [1.0, 0.0, 0.0];
        // var p2 = [0.0, 0.0, 1.0];
        // this.drawPredictPath([p1, p2]);
    }

    parseDataUpdate(msg) {
        super.parseDataUpdate(msg);
        switch (msg["name"]) {
            case "prediction":
                this.onUpdatePrediction();
                break;
            case "allPairsPrediction":
                this.onUpdateAllPairPrediction();
                break;
            case "currentPair":
                this.clearUI();
                console.log(this.data['currentPair']['sentences']);
                this.onUpdateGroundTruth(this.data['currentPair']["label"]);
                break;
            case "predictionUpdate":
                let pred = this.data["predictionUpdate"];
                this.onUpdateOptimizedPrediction(pred);
                break;
            case "predictionBatchUpdate":
                let preds = this.data["predictionBatchUpdate"];
                this.onUpdateOptimizedBatchPrediction(preds);
        }
    }

    clearUI() {
        if (this.svg) {
            // this.onUpdateGroundTruth("");
            this.svg.select(this.div + "overlay").remove();
            this.svg.selectAll(".predCircle").remove();
            this.svg.selectAll(".predSquare").remove();
            this.svg.selectAll(".dotPredPath").remove();
            this.svg.selectAll(".predPath").remove();
        }
    }

    resize() {
        // console.log("prediction-resize\n");
        this.draw();
    }

    pred2Pos(d) {
        let x = d[1] * 112 + d[2] * 0 + d[0] * 224;
        let y = d[1] * 0 + d[2] * 194 + d[0] * 194;
        return [x, y];
    }

    //trigger request to reassign prediction
    onPredictionReassign(label) {

        let pipeline = this.data["pipeline"];
        // console.log(pipeline);
        let learningRate = this.data["learningRate"];
        let C_mira = this.data["C_mira"];
        let iteration = this.data["iteration"];
        //call python side
        console.log(learningRate, C_mira, iteration);
        this.callFunc("predictUpdate", {
            "newLabel": label,
            "iteration": iteration,
            "learningRate": learningRate,
            "encoderFlag": pipeline[0]["state"],
            "attFlag": pipeline[1]["state"],
            "classFlag": pipeline[2]["state"],
            "mira_c": C_mira
        })
    }

    //trigger when python return optimized
    onUpdateOptimizedPrediction(predictionUpdate) {
        // console.log(predictionUpdate, this.selectPred);
        this.drawPredictPath([this.selectPred, predictionUpdate], "solid");
    }

    onUpdateOptimizedBatchPrediction(preds) {
        // console.log(preds);
        //draw batch prediction
        this.updatePredictDisplay();
        this.svg.selectAll(".predSquare").remove();
        this.svg.selectAll(".predSquare")
            .data(preds)
            .enter()
            .append("rect")
            .attr("class", "predSquare")
            .attr("id", (d, i) => {
                return "rect" + i;
            })
            .attr("x", d => {
                return d[1] * 112 + d[2] *
                    0 +
                    d[0] * 224 - 3;
            })
            .attr("y", d => {
                return d[1] * 0 + d[2] *
                    194 +
                    d[0] * 194 - 3;
            })
            .attr("width", 6)
            .attr("height", 6)
            .style("fill", "lightblue")
            .style("stroke", "white")
            .style("opacity", 1.0)
            .on("click", (d, i) => {
                this.callFunc("updatePipelineStateFromIndex", {
                    "index": i
                });
            })

        for (var i = 0; i < preds.length; i++) {
            this.drawPredictPath([this.selectPred, preds[i]],
                "solid", false);
        }

    }

    onUpdateGroundTruth(label) {
        // console.log(label);
        if (label === "neutral") {
            this.Neutral.style("fill", "lightgreen");
            this.Contradiction.style("fill", "none");
            this.Entailment.style("fill", "none");
            // this.Contradiction.style("fill", "lightgreen");
            // this.Entailment.style("fill", "lightgreen");
        } else if (label === "contradiction") {
            this.Neutral.style("fill", "none");
            this.Contradiction.style("fill", "lightgreen");
            this.Entailment.style("fill", "none");
        } else if (label === "entailment") {
            this.Neutral.style("fill", "none");
            this.Contradiction.style("fill", "none");
            this.Entailment.style("fill", "lightgreen");
        } else {
            this.Neutral.style("fill", "none");
            this.Contradiction.style("fill", "none");
            this.Entailment.style("fill", "none");
        }
    }

    onUpdatePrediction() {
        //cleanup
        this.drawPredictPath();
        this.drawDensityOverlay([]);

        //clone prediction
        var prediction = this.data['prediction'][0].slice(0);
        //add sentence index
        prediction.concat([0, 0]);
        //reverse prediction
        // prediction = prediction.reverse();
        // console.log(prediction);
        this.updatePredictDisplay([prediction]);
    }

    onUpdateAllPairPrediction() {

        var data = [];
        var allPairsPrediction = this.data["allPairsPrediction"].slice(0);
        // allPairsPrediction = allPairsPrediction.reverse();
        // console.log(allPairsPrediction);

        //the euclidean coordinate data
        var dataXY = [];
        for (var i = 0; i < allPairsPrediction.length; i++) //per source
            for (var j = 0; j < allPairsPrediction[i].length; j++) { // per target
            if (i === 0 || j === 0) {
                data.push(allPairsPrediction[i][j].concat([i, j]));
                let d = allPairsPrediction[i][j];
                let x = d[1] * 112 + d[2] * 0 + d[0] * 224;
                let y = d[1] * 0 + d[2] * 194 + d[0] * 194;
                dataXY.push([x, y]);
            }
        }

        // console.log(data);
        this.drawDensityOverlay(dataXY)
        this.updatePredictDisplay(data);
    }

    updatePredictDisplay(data) {
        // console.log(this.data);
        // neutral, Contradiction, Entailment
        // Entailment, neutral, contradiction
        // (112,0) (0,194) (224,194)
        this.svg.selectAll(".predCircle").remove();
        this.svg.selectAll(".predSquare").remove();
        if (data !== undefined) {
            // console.log(data);
            var pLen = data.length;
            if (pLen > 1) {
                data = JSON.parse(JSON.stringify(data));
                data = data.reverse();
            }

            this.svg.selectAll(".predCircle")
                .data(data)
                .enter()
                .append("circle")
                .attr("class", "predCircle")
                .attr("id", (d, i) => {
                    return "circle" + i;
                })
                .attr("cx", d => {
                    return d[1] * 112 + d[2] *
                        0 +
                        d[0] * 224;
                })
                .attr("cy", d => {
                    return d[1] * 0 + d[2] *
                        194 +
                        d[0] * 194;
                })
                .attr("r", (d, i) => {
                    if (i === pLen - 1) return 6;
                    else return 3;
                })
                .style("fill", (d, i) => {
                    if (i === pLen - 1) {
                        return 'gold';
                    } else {
                        return 'grey';
                    }
                })
                .style("stroke", "white")
                .style("opacity", 0.8)
                .on("mouseover", function(d) {

                })
                .on("mouseout", function(d) {

                })
                //.style("opacity", (d,i)=>{if (i==0) return "1.0"; else return "0.5";})
                .on("click", (d, i) => {
                    // console.log(d);
                    var source, target;
                    if (this.data["allSourceSens"])
                        source = this.data["allSourceSens"][d[3]];
                    else
                        source = this.data["currentPair"]["sentences"][
                            0
                        ];

                    if (this.data["allTargetSens"])
                        target = this.data["allTargetSens"][d[4]];
                    else
                        target = this.data["currentPair"]["sentences"][
                            1
                        ];

                    console.log("currentPair", source, target, this.data[
                        "currentPair"]["sentences"]);
                    this.data["currentPair"]["sentences"] = [
                        source,
                        target
                    ];
                    //update the pair
                    // console.log("update pair/att", this.data[
                    // "allSourceSens"]);
                    this.setData("currentPair", this.data[
                        "currentPair"]);

                    //then update the current attention
                    this.callFunc("attention");

                })
                .call(d3.drag()
                    .on("start", this.dragstarted.bind(this))
                    .on("drag", this.dragged.bind(this))
                    .on("end", this.dragended.bind(this)));
        }
    }

    //triangle range: 224, 194
    drawDensityOverlay(dataPoints) {
        this.svg.select(this.div + "overlay").remove();
        if (dataPoints.length > 1) {
            this.svg.append("g")
                .attr("id", this.uuid + "overlay")
                .attr("fill", "none")
                .attr("stroke", "grey")
                .attr("stroke-width", 0.8)
                .attr("stroke-linejoin", "round")
                .selectAll("path")
                .data(d3.contourDensity()
                    .x(function(d) {
                        return d[0];
                    })
                    .y(function(d) {
                        return d[1];
                    })
                    .size([224, 194])
                    .bandwidth(4)
                    .thresholds(12)
                    (dataPoints))
                .enter().append("path")
                .attr("clip-path", "url(#triClip)")
                .attr("opacity", 0.5)
                .attr("d", d3.geoPath());
        }
    }

    /////////////// drag ////////////////

    dragstarted(d) {
        // console.log("dragStarted:", d);
    }

    dragged(d) {
        var pos = this.pred2Pos(d);

        if (this.svg.select("#reassignPredict").empty()) {
            // console.log("dragged:", d);
            // var pos = [0, 0];
            var w = 20;
            var h = 15;

            var g = this.svg.append("g").attr("id",
                "reassignPredict");

            var entailRect = g.append("g");

            var neutralRect = g.append("g");
            var contractRect = g.append("g");
            var that = this;

            this.selectPred = d;

            ///////// N ///////////
            neutralRect.append("rect")
                .attr("x", pos[0] - w / 2)
                .attr("y", pos[1] - 20 - h / 2)
                .attr("width", w)
                .attr("height", h)
                .attr("fill", "lightgrey")
                .attr("stroke", "black")
                .on("mouseover", function(d) {
                    d3.select(this).attr("fill", "grey");
                    that.reassignedPred = [0, 1, 0];
                    that.drawCurrentAssignedPred();
                })
                .on("mouseout", function(d) {
                    d3.select(this).attr("fill",
                        "lightgrey");
                    that.reassignedPred = undefined;
                    that.drawCurrentAssignedPred();
                })
            neutralRect.append("text")
                .attr("x", pos[0])
                .attr("y", pos[1] - 20)
                .attr("dy", ".35em")
                .attr("font-size", 10)
                .text("N")
                .style("text-anchor", "middle")
                .style("pointer-events", "none");

            ///////// E ///////////
            entailRect.append("rect")
                .attr("x", pos[0] + 20 - w / 2)
                .attr("y", pos[1] + 15 - h / 2)
                .attr("width", w)
                .attr("height", h)
                .attr("fill", "lightgrey")
                .attr("stroke", "black")
                .on("mouseover", function(d, i) {
                    d3.select(this).attr("fill", "grey");
                    that.reassignedPred = [1, 0, 0];
                    that.drawCurrentAssignedPred();
                })
                .on("mouseout", function(d) {
                    d3.select(this).attr("fill",
                        "lightgrey");
                    that.reassignedPred = undefined;
                    that.drawCurrentAssignedPred();
                })

            entailRect.append("text")
                .attr("x", pos[0] + 20)
                .attr("y", pos[1] + 15)
                .attr("dy", ".35em")
                .attr("font-size", 10)
                .text("E")
                .style("text-anchor", "middle")
                .style("pointer-events", "none");

            ///////// C ///////////
            contractRect.append("rect")
                .attr("x", pos[0] - 20 - w / 2)
                .attr("y", pos[1] + 15 - h / 2)
                .attr("width", w)
                .attr("height", h)
                .attr("fill", "lightgrey")
                .attr("stroke", "black")
                .on("mouseover", function(d) {
                    d3.select(this).attr("fill", "grey");
                    that.reassignedPred = [0, 0, 1];
                    that.drawCurrentAssignedPred();
                })
                .on("mouseout", function(d) {
                    d3.select(this).attr("fill",
                        "lightgrey");
                    that.reassignedPred = undefined;
                    that.drawCurrentAssignedPred();
                })

            contractRect.append("text")
                .attr("x", pos[0] - 20)
                .attr("y", pos[1] + 15)
                .attr("dy", ".35em")
                .attr("font-size", 10)
                .text("C")
                .style("text-anchor", "middle")
                .style("pointer-events", "none");
        } else {
            //draw line from the center
            // var currentPos = [d3.event.x, d3.event.y];
            // this.svg.selectAll(".dragline").remove();
            // this.svg
            //     .append("line")
            //     .attr("class", "dragline")
            //     .attr("x1", pos[0])
            //     .attr("y1", pos[1])
            //     .attr("x2", currentPos[0])
            //     .attr("y2", currentPos[1])
            //     .attr("stroke", "grey");
            // console.log("dragged:", currentPos);
        }
    }

    dragended(d) {
        // console.log("dragended", this.reassignedPred);
        //check the location
        this.svg.select("#reassignPredict").remove();
        //trigger optimizaton on the python side
        if (this.reassignedPred) {
            var i = this.reassignedPred.indexOf(Math.max(...this
                .reassignedPred));
            this.onPredictionReassign(i);
            //reset reassignPred so it won't be trigger when click on the point
            this.reassignedPred = undefined;
        }
    }

    drawCurrentAssignedPred() {
        if (this.reassignedPred) {
            this.drawPredictPath([
                this.selectPred,
                this.reassignedPred
            ], "dotted");

        } else {
            this.drawPredictPath();
        }
    }

    //drawing a series of predictions
    //the old prediction circle with dotted line
    //the new prediction is solid
    drawPredictPath(path, type = "dotted", clearPrevious = true) {
        if (path === undefined) {
            this.svg.selectAll(".dotPredPath").remove();
            this.svg.selectAll(".predPath").remove();
        } else {
            var line = [this.pred2Pos(path[0]), this.pred2Pos(
                path[1])];
            // console.log(line);

            //draw arrow
            var d3line = d3.line()
                .x(function(d) {
                    return d[0];
                })
                .y(function(d) {
                    return d[1];
                });

            if (type === "dotted") {
                this.svg.selectAll(".dotPredPath").remove();

                this.svg.append("circle")
                    .attr("class", "dotPredPath")
                    .attr("cx", line[1][0])
                    .attr("cy", line[1][1])
                    .attr("r", 6)
                    .style("stroke-dasharray", ("2, 2"))
                    .style('stroke', 'grey')
                    .style("fill", "none");

                this.svg.append('path')
                    .attr("class", "dotPredPath")
                    .attr('fill', '#999')
                    .style("stroke-dasharray", ("2, 2"))
                    .style('stroke', 'grey')
                    .attr("marker-end", "url(#arrowhead)")
                    .attr("d", d => d3line(line));
            } else if (type === "solid") {
                // console.log("draw path line");
                if (clearPrevious) {
                    this.svg.selectAll(".predCircle").remove();
                    this.svg.selectAll(".predPath").remove();
                }

                this.svg.append("circle")
                    .attr("class", "predPath")
                    .attr("cx", line[0][0])
                    .attr("cy", line[0][1])
                    .attr("r", 6)
                    // .style("stroke-dasharray", ("2, 2"))
                    .style('stroke', 'grey')
                    .style("fill", "none");

                // this.svg.append("circle")
                //     .attr("class", "predPath")
                //     .attr("cx", line[1][0])
                //     .attr("cy", line[1][1])
                //     .attr("r", 6)
                //     // .style("stroke-dasharray", ("2, 2"))
                //     .style('stroke', 'white')
                //     .attr("fill", "grey");

                this.svg.append('path')
                    .attr("class", "predPath")
                    .attr('fill', '#999')
                    .style('stroke', 'grey')
                    // .attr("marker-end", "url(#arrowhead)")
                    // .attr("d", d => d3line(line));

                //update prediction
                if (clearPrevious) {
                    var prediction = path[1];
                    //add sentence index
                    prediction.concat([0, 0]);
                    this.updatePredictDisplay([prediction]);
                }
            }
        }
    }
}
