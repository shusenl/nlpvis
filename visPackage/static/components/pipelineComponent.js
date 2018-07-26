class pipelineComponent extends baseComponent {
    constructor(uuid) {
        super(uuid);

        this.subscribeDatabyNames(["pipeline"]);

        this.margin = {
            top: 10,
            right: 10,
            bottom: 10,
            left: 10
        };

        this.draw();
    }

    initSvg() {
        //create svg
        if (this.svgContainer === undefined) {
            this.svgContainer = d3.select(this.div).append("svg")
                .attr("width", this.pwidth)
                .attr("height", this.pheight);
            //add stripe pattern
            this.defs = this.svgContainer.append("defs");
            this.defs.append("pattern")
                .attr("id", "stripe")
                .attr("patternUnits", "userSpaceOnUse")
                .attr("width", 20)
                .attr("height", 20)
                .attr("patternTransform", "rotate(45)")
                .append("line")
                .attr("x1", 15)
                .attr("y1", 0)
                .attr("x2", 15)
                .attr("y2", 20)
                .attr("stroke", "lightgrey")
                .attr("stroke-width", 10);

            this.defs.append("marker")
                .attr("id", "arrow")
                .attr("markerWidth", 5)
                .attr("markerHeight", 5)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 5)
                .attr("refY", 0)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .style("fill", "grey");

            //draw legend
            this.legend = this.svgContainer.append("g").attr("id", "legend");

            this.drawConfigurationOption();
            this.drawResetBotton();

            this.svg = this.svgContainer
                .append("g")
                .attr("transform", "translate(" + this.margin.left + "," +
                    this.margin.top + ")");

            this.svgSave = new svgExporter(this.svgContainer, [this.width -
                10, 10
            ]);
        } else {
            this.svgContainer
                .attr("width", this.pwidth)
                .attr("height", this.pheight)

            this.svg.selectAll("text,rect,path").remove();

            this.svgSave.updatePos([this.width - 10, 10])
            this.svgSave.draw();
        }
    }

    drawResetBotton() {
        //reset
        if (this.svgContainer.select("#resetButton").empty())
            this.reset = this.svgContainer.append("g").attr("id",
                "resetButton");
        else
            this.svgContainer.select("#resetButton").selectAll("*")
            .remove();

        //
        this.Cslider = new sliderPlot(this.mode, [50, 50], [70, 15],
            "C", [1.0, 10.0], 2.0, ".0f");
        this.Cslider.bindUpdateCallback(d => {
            this.setData("C_mira", Number(this.Cslider.value));
        });

        this.learnRate = new sliderPlot(this.mode, [50, 70], [70, 15],
            "l_rate", [0.000001, 0.0001], 0.00001, ".4f");
        this.learnRate.bindUpdateCallback(d => {
            this.setData("learningRate", Number(this.learnRate.value));
        });

        this.iteration = new sliderPlot(this.mode, [50, 90], [70, 15],
            "iter", [5, 20], 15, ".0f");
        this.iteration.bindUpdateCallback(d => {
            this.setData("iteration", Number(this.iteration.value));
        });

        this.reset.append("rect")
            .attr("rx", 3)
            .attr("ry", 3)
            .attr("x", 10)
            .attr("y", 10)
            .attr("width", 100)
            .attr("height", 30)
            .attr("fill", "lightgrey")
            .on("click", this.resetPipeline.bind(this))
            .on("mouseover", function(d) {
                d3.select(this).attr("fill", "grey");
            })
            .on("mouseout", function(d) {
                d3.select(this).attr("fill", "lightgrey");

            });
        this.reset.append("text")
            .text("reset model")
            .attr("x", 10 + 50)
            .attr("y", 10 + 15)
            .style("text-anchor", "middle")
            .style("alignment-baseline", "middle")
            .style("pointer-events", "none");
    }

    drawConfigurationOption() {
        if (this.svgContainer.select("#configurationOption").empty())
            this.mode = this.svgContainer.append("g").attr("id",
                "configurationOption");
        else
            this.svgContainer.select("#configurationOption").selectAll("*")
            .remove();

        this.updateMode = [{
            "name": "current configuration",
            "mode": "single",
            "on": true
        }, {
            "name": "all configurations",
            "mode": "batch",
            "on": false
        }];

        //set default mode
        this.setData("updateMode", "single");
        let that = this;
        this.mode.selectAll(".modeSelector")
            .data(this.updateMode)
            .enter()
            .append("rect")
            .attr("class", "modeSelector")
            // .attr("rx", 3)
            // .attr("ry", 3)
            .attr("x", (d, i) => this.width * 0.5 + i * 160 - 160)
            .attr("y", this.height - 40)
            .attr("width", 160)
            .attr("height", 30)
            .attr("fill", d => {
                if (d.on)
                    return "lightblue";
                else
                    return "white";
            })
            .style("stroke", "lightblue")
            .style("stroke-width", 2)
            .on("click", function(d, i) {
                //reset the mode
                that.mode.selectAll(".modeSelector").attr("fill",
                    "white");
                that.updateMode.map(d => {
                    d.on = false
                });

                //set the current
                d3.select(this).attr("fill", "lightblue");
                that.updateMode[i].on = true;
                that.setData("updateMode", d.mode);
            })


        this.mode.selectAll(".modeSelectorLabel")
            .data(this.updateMode)
            .enter()
            .append("text")
            .attr("class", "modeSelectorLabel")
            .text(d => d.name)
            .attr("x", (d, i) => this.width * 0.5 + i * 160 -
                80)
            .attr("y", this.height - 40 + 15)
            .style("text-anchor", "middle")
            .style("alignment-baseline", "middle")
            .style("pointer-events", "none");
    }

    resetPipeline() {
        console.log("reset model");
        let pipeline = this.data["pipeline"];
        for (let i = 0; i < pipeline.length; i++) {
            delete pipeline[i]["hist"];
            pipeline[i]['state'] = true;
        }
        // console.log(pipeline);
        this.setData("pipeline", pipeline);
        this.draw();
        this.callFunc("reloadModel");
    }

    resize() {
        this.draw();
    }

    parseDataUpdate(msg) {
        super.parseDataUpdate(msg);
        switch (msg["name"]) {
            case "pipeline":
                let states = this.data["pipelineState"];
                // console.log(states);
                this.draw();
                break
        }
    }

    updatePipelineState(index, state) {
        let pipeline = this.data["pipeline"];
        pipeline[index]["state"] = state;
        // console.log("updatePipeline: ", index, state, pipeline);
        this.setData("pipeline", pipeline);
    }

    draw() {
        this._updateWidthHeight();
        // console.log("draw pipeline");
        if (this.data["pipeline"] !== undefined) {
            this.initSvg();
            this.drawLegend();

            this.items = [];
            var pipelineData = this.data["pipeline"];
            // console.log(pipelineData);

            // if (this.items === undefined) {
            var len = pipelineData.length;
            var size = [this.width / (len + 1), 45];
            for (var i = 0; i < pipelineData.length; i++) {
                var pos = [this.width / (len) * (i + 0.5), this.height *
                    0.5
                ];
                var item = new pipelineItemPlot(this.svg,
                    pos, size, pipelineData[i]["index"], pipelineData[i]
                    ["name"], pipelineData[i]["state"]
                );
                item.bindSelectionCallback(this.updatePipelineState.bind(
                    this));
                if (pipelineData[i]["hist"]) {
                    item.setGraidentHisto(pipelineData[i]["hist"],
                        pipelineData[i]["histName"]);
                }
                item.draw();
                this.items.push(item);
            }
            for (var i = 0; i < pipelineData.length; i++) {
                let arrows = pipelineData[i]["arrow"];
                for (let j = 0; j < arrows.length; j++) {
                    let start = this.items[i].getOutputPortPos();
                    let end = this.items[arrows[j]].getInputPortPos();
                    this.drawArrow(start, end);
                }
            }
            // }
            // else {
            //     //update pipeline
            //     for(let i=0; i<this.items.length; i++){
            //         this.items[i]
            //     }
            // }
        }
    }

    drawArrow(start, end) {
        // console.log("drawArrow", start, end)
        //align the arrow head with rect
        end[0] = end[0] - 4;
        //draw curved arrow
        let start1 = [start[0] + (end[0] - start[0]) * 0.2, start[1]];
        let end1 = [start[0] + (end[0] - start[0]) * 0.8, end[1]];
        var pathData = [start, start1, end1, end];
        let curveGen = d3.line()
            // .curve(d3.curveCatmullRomOpen)
            .curve(d3.curveMonotoneX)
            .x(d => d[0])
            .y(d => d[1]);

        this.svg.append("path")
            .attr("d", curveGen(pathData))
            .attr("stroke-width", 2)
            .attr("stroke", "grey")
            .attr("fill", "none")
            .attr("marker-end", "url(#arrow)");
    }

    drawLegend() {
        this.legend.selectAll("*").remove();
        this.legend.append("rect")
            .attr("fill", "lightblue")
            .attr("stroke", "grey")
            .attr("x", this.width - 90)
            .attr("y", 10)
            .attr("width", 70)
            .attr("height", 30);
        this.legend.append("text")
            .attr("x", this.width - 100)
            .attr("y", 25)
            .attr("font-size", 14)
            .text("Allow Update")
            .style("text-anchor", "end")
            .style("alignment-baseline", "middle");

        this.legend.append("rect")
            .attr("fill", "url(#stripe)")
            .attr("stroke", "grey")
            .attr("x", this.width - 90)
            .attr("y", 50)
            .attr("width", 70)
            .attr("height", 30);
        this.legend.append("text")
            .attr("x", this.width - 100)
            .attr("y", 65)
            .attr("font-size", 14)
            .text("Freeze Update")
            .style("text-anchor", "end")
            .style("alignment-baseline", "middle");
    }
}
