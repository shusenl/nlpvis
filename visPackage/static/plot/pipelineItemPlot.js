class pipelineItemPlot {
    constructor(svg, pos, size, index, label, state) {
        this.svg = svg.append("g");
        this.pos = pos;
        this.size = size;
        this.label = label;
        this.index = index;
        //store whether the layer can be updated
        this.updateFlag = state;
    }

    setState(state) {
        this.state = state;
        this.draw();
    }

    setGraidentHisto(histo, histName) {
        if (histo) {
            this.histoList = histo;
            this.histName = histName;
        }
    }

    clear() {
        this.histoList = undefined;
        // this.histName = undefined;
    }

    bindSelectionCallback(callback) {
        this.callback = callback;
    }

    draw() {
        // console.log("draw pipeline item");
        let that = this;
        if (this.svg.select("rect").empty()) {
            this.svg.append("rect")
                .attr("x", this.pos[0] - this.size[0] * 0.5)
                .attr("y", this.pos[1] - this.size[1] * 0.5)
                .attr("width", this.size[0])
                .attr("height", this.size[1])
                .attr("fill", d => {
                    if (this.updateFlag)
                        return "lightblue";
                    else
                        return "url(#stripe)"
                })
                .attr("stroke", "lightgrey")
                .on("click", function() {
                    // console.log(d3.select(this).attr("fill"));
                    if (d3.select(this).attr("fill") ===
                        "lightblue") {
                        that.updateFlag = false;
                        d3.select(this).attr("fill",
                            "url(#stripe)");
                        that.callback(that.index, that.updateFlag);
                    } else {
                        that.updateFlag = true;
                        d3.select(this).attr("fill",
                            "lightblue");
                        that.callback(that.index, that.updateFlag);
                    }
                });

            /////// title ///////
            this.svg.append("rect")
                .attr("x", this.pos[0] - this.size[0] * 0.5)
                .attr("y", this.pos[1] - this.size[1] * 1.5)
                .attr("width", this.size[0])
                .attr("height", this.size[1])
                .attr("fill", "grey");

            this.svg.append("text")
                .attr("x", this.pos[0])
                .attr("y", this.pos[1] - this.size[1] * 1.0 + 5)
                .text(this.label)
                .attr("fill", "white")
                .style("text-anchor", "middle")
                .style("pointer-events", "none");

            let hiddenLayerBoxPos = [this.pos[0] - this.size[0] * 0.5,
                this.pos[1] + this.size[1] * 0.5
            ];
            let hiddenLayerBoxSize = [this.size[0],
                this.size[1] * 2
            ];

            this.svg.append("rect")
                .attr("x", hiddenLayerBoxPos[0])
                .attr("y", hiddenLayerBoxPos[1])
                .attr("width", hiddenLayerBoxSize[0])
                .attr("height", hiddenLayerBoxSize[1])
                .attr("fill", "white")
                .attr("stroke", "lightgrey");

            this.svg.append("text")
                .attr("x", this.pos[0])
                .attr("y", this.pos[1] + 5)
                .text("parameters")
                .style("text-anchor", "middle")
                .style("pointer-events", "none");

            this.svg.append("text")
                .attr("x", this.pos[0])
                .attr("y", this.pos[1] + 2.30 * this.size[1])
                .text(this.histName)
                .style("font-size", 12)
                .style("text-anchor", "middle")
                .style("pointer-events", "none");

            //create histogram to disable distribution of value update
            this.hist = new histoPlot(this.svg, [
                hiddenLayerBoxPos[0] + 4,
                hiddenLayerBoxPos[1] + 4
            ], [
                hiddenLayerBoxSize[0] - 8,
                hiddenLayerBoxSize[1] - 25
            ], true);
            if (this.histoList) {
                this.hist.setHisto(this.histoList);
            }

        }
        // else {
        //     this.svg.select("rect")
        //         .attr("x", this.pos[0] - this.size[0] * 0.5)
        //         .attr("y", this.pos[1] - this.size[1] * 0.5)
        //         .attr("width", this.size[0])
        //         .attr("height", this.size[1]);
        //     this.hist.update()
        // }
    }

    getOutputPortPos() {
        return [
            this.pos[0] + this.size[0] * 0.5,
            this.pos[1] + this.size[1] * 1.5
        ]
    }

    getInputPortPos() {
        return [
            this.pos[0] - this.size[0] * 0.5,
            this.pos[1]
        ]
    }
}
