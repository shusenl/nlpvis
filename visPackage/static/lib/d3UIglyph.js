class d3UIplotGlyph {
    constructor(svg, pos = [0, 0], size = [60, 60], histOfSetIndex = 1, histo =
        true) {

        this.histOfSetIndex = histOfSetIndex;
        this.histFlag = histo;
        // this.svgTag = "#" + svgTag;
        this.svgContainer = svg;
        this._pos = pos;
        this._size = size;
        // console.log(list);
        // this._list = JSON.parse(JSON.stringify(list)); //list to select from
        this.id = 'd3UIglyph-' + uuidv1();
    }

    setData(plotData, valueData, colorMap, hist, names = undefined, evidence =
        0.0) {
        this.names = names;
        this.plotData = plotData;
        this.valueData = valueData;
        this.evidence = evidence;
        // console.log(this.evidence);

        var cmrange = [];
        var cmlen = colorMap.length;
        var range = d3.extent(valueData);
        for (var i = 0; i < cmlen; i++) {
            cmrange.push(range[0] + (i / (cmlen - 1)) * (range[
                1] - range[0]));
        }
        this.colorMap = d3.scale.linear()
            .domain(cmrange)
            .range(colorMap);
        // console.log(this.colorMap(0.5));

        // console.log(plotData, d3.extent(plotData.map(d => d[0])), d3.extent(
        // plotData.map(d => d[1])));

        this.scaleX = d3.scale.linear().domain(
            d3.extent(plotData.map(d => d[0])));
        this.scaleY = d3.scale.linear().domain(
            d3.extent(plotData.map(d => d[1])));

        //init histogram data
        this.histData = hist;
        this.histScaleX = d3.scale.linear().domain(
            d3.extent([0, hist.length]));
        hist.push(0);
        this.histScaleY = d3.scale.linear().domain(
            d3.extent(hist));

        this.draw();
    }

    draw() {
        if (this.svgContainer.select("#fobj_" + this.id).empty()) {

            this.foreignObject = this.svgContainer.append("foreignObject")
                .attr("id", "fobj_" + this.id)
                .attr("x", this._pos[0] - this._size[0] * 0.5)
                .attr("y", this._pos[1] - this._size[1] * 0.5)
                .attr("width", this._size[0] * 2)
                .attr("height", this._size[1])
                .attr("pointer-events", "none");

            // add embedded body to foreign object
            this.foBody = this.foreignObject.append("xhtml:body")
                .attr("margin", 0)
                .attr("padding", 0)
                .attr("background-color", "white")
                .attr("width", this._size[0])
                .attr("height", this._size[1])
                // .style("border", "1px solid lightgray")
                .attr("pointer-events", "none");

            // add embedded canvas to embedded body
            this.canvas = this.foBody.append("canvas")
                .attr("x", 0)
                .attr("y", 0)
                .attr("id", "canvas_" + this.id)
                .attr("width", this._size[0])
                .attr("height", this._size[1])
                .style("cursor", "crosshair")
                .attr("pointer-events", "none");

            if (this.evidence !== 0.0) {
                this.svgContainer.append("text")
                    .attr("id", "evidence" + this.id)
                    .attr("x", this._pos[0] - 0.5 * this._size[0])
                    .attr("y", this._pos[1] - 0.5 * this._size[1] - 5)
                    .text(d3.format(".4f")(this.evidence));
            }

            if (this.names !== undefined) {
                this.svgContainer
                    .append("text")
                    .attr("id", "AxisLabel2" + this.id)
                    .attr("transform", "translate(" + Number(this._pos[0] +
                        0.5 *
                        this._size[0] + 12) + "," + Number(this._pos[1] +
                        0.5 * this._size[1]) + ") rotate(-90)")
                    .text(this.names[1])
                this.svgContainer.append("text")
                    .attr("id", "AxisLabel1" + this.id)
                    .attr("x", this._pos[0] - 0.5 * this._size[0])
                    .attr("y", this._pos[1] + 0.5 * this._size[1] + 12)
                    .text(this.names[0]);
            }

            /////////////// histogram ///////////////
            if (this.histFlag) {
                this.histForeignObject = this.svgContainer.append(
                        "foreignObject")
                    .attr("id", "histfobj_" + this.id)
                    .attr("x", d => {
                        if (this.histOfSetIndex > 0) {
                            return this._pos[0] + this._size[0] * 0.5 +
                                15;
                        } else {
                            return this._pos[0] - this._size[0] * 1.5 -
                                15;
                        }
                    })
                    .attr("y", this._pos[1] - this._size[1] * 0.5)
                    .attr("width", this._size[0])
                    .attr("height", this._size[1])
                    // .attr("pointer-events", "none");

                this.histBody = this.histForeignObject.append("xhtml:body")
                    .attr("margin", 0)
                    .attr("padding", 0)
                    .attr("background-color", "red")
                    .attr("width", this._size[0])
                    .attr("height", this._size[1])
                    .style("border", "1px solid lightgray")
                    // .attr("pointer-events", "none");

                this.histCanvas = this.histBody.append("canvas")
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("id", "histcanvas_" + this.id)
                    .attr("width", this._size[0])
                    .attr("height", this._size[1])
                    .style("cursor", "crosshair")
                    // .attr("pointer-events", "none");
            }

        } else {
            if (this.names !== undefined) {
                this.svgContainer.select("#AxisLabel1" + this.id)
                    .attr("x", this._pos[0] - 0.5 * this._size[0])
                    .attr("y", this._pos[1] + 0.5 * this._size[1] + 12);
                this.svgContainer.select("#AxisLabel2" + this.id)
                    .attr("id", "AxisLabel2" + this.id)
                    .attr("transform", "translate(" + Number(this._pos[0] +
                        0.5 *
                        this._size[0] + 12) + "," + Number(this._pos[1] +
                        0.5 * this._size[1]) + ") rotate(-90)");
            }

            if (this.evidence !== 0.0) {
                this.svgContainer.select("#evidence" + this.id)
                    .attr("x", this._pos[0] - 0.5 * this._size[0])
                    .attr("y", this._pos[1] - 0.5 * this._size[1] - 5);
            }
            //update group
            this.foreignObject
                .attr("x", this._pos[0] - this._size[0] * 0.5)
                .attr("y", this._pos[1] - this._size[1] * 0.5)
                .attr("width", this._size[0])
                .attr("height", this._size[1]);

            this.foBody
                .attr("width", Math.round(this._size[0]))
                .attr("height", Math.round(this._size[1]));

            // add embedded canvas to embedded body
            this.canvas
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", Math.round(this._size[0]))
                .attr("height", Math.round(this._size[1]));

            //////////// histogram ////////////
            if (this.histFlag) {
                this.histForeignObject.attr("x", d => {
                        if (this.histOfSetIndex > 0) {
                            return this._pos[0] + this._size[0] * 0.5 +
                                15;
                        } else {
                            return this._pos[0] - this._size[0] * 1.5 -
                                15;
                        }
                    })
                    .attr("y", this._pos[1] - this._size[1] * 0.5)
                    .attr("width", this._size[0])
                    .attr("height", this._size[1]);

                this.histBody
                    .attr("width", this._size[0])
                    .attr("height", this._size[1]);

                this.histCanvas
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("width", Math.round(this._size[0]))
                    .attr("height", Math.round(this._size[1]));
            }

            fabric.Object.prototype.transparentCorners = false;
        }

        //draw plot
        this.fCanvas = new fabric.StaticCanvas("canvas_" + this
            .id, {
                // backgroundColor: "#000",
                renderOnAddRemove: false
            });

        this.scaleX.range([0, 0.95 * this._size[0]]);
        this.scaleY.range([0, 0.95 * this._size[1]]);

        for (var i = 0; i < this.plotData.length; i++) {
            var dot = new fabric.Circle({
                left: this.scaleX(this.plotData[i][0]),
                top: this._size[1] * 0.95 - this.scaleY(this.plotData[
                    i][1]),
                radius: this._size[0] / 70.0,
                fill: this.colorMap(this.valueData[i]),
                objectCaching: false
            });
            this.fCanvas.add(dot);
        }
        // var label1 = new fabric.Text(this.names[0], {
        //     left: 0.2 * this._size[0],
        //     top: 0.01 * this._size[1],
        //     fontSize: 15
        // });
        // var label2 = new fabric.Text(this.names[1], {
        //     left: 0.99 * this._size[0],
        //     top: 0.2 * this._size[1],
        //     angle: 90,
        //     fontSize: 15
        // });
        //
        // this.fCanvas.add(label1)
        // this.fCanvas.add(label2)

        this.fCanvas.renderAll();

        this.drawHisto();
    }

    drawHisto() {
        // var testData = [1, 2.5, 3, 4.1, 1, 2, 3, 3, 1.2, 4];
        // this.histogram(testData);
        if (this.histData !== undefined && this.histFlag) {
            //histogram
            this.histScaleX.range([0, this._size[0]]);
            this.histScaleY.range([0, this._size[1]]);

            this.fhistCanvas = new fabric.StaticCanvas("histcanvas_" + this
                .id, {
                    // backgroundColor: "#000",
                    renderOnAddRemove: false
                });

            for (var i = 0; i < this.histData.length; i++) {
                var rect = new fabric.Rect({
                    left: this.histScaleX(i),
                    top: this._size[1] - this.histScaleY(this.histData[
                        i]),
                    width: this.histScaleX(1),
                    height: this.histScaleY(this.histData[i]),
                    fill: "blue",
                    objectCaching: false
                });
                this.fhistCanvas.add(rect);
            }
            this.fhistCanvas.renderAll();
        }
    }

    drawPopup() {

    }

    updateHistogram(hist) {
        this.histData = hist;
        this.histScaleX = d3.scale.linear().domain(
            d3.extent([0, hist.length]));
        hist.push(0);
        this.histScaleY = d3.scale.linear().domain(
            d3.extent(hist));
        this.drawHisto();
    }

    size(size) {
        this._size = size;
        this.draw();
    }

    pos(pos) {
        this._pos = pos;
        this.draw();
    }

    update(pos, size) {
        this._pos = pos;
        this._size = size;
        this.draw();
    }

    getInputPortCoord() {
        return [this._pos[0] - this._size[0] * 0.5, this._pos[1]];
    }

    getOutputPortCoord() {
        return [this._pos[0] + this._size[0] * 0.5, this._pos[1]];
    }

    setClickCallback(func) {
        this.clickCallback = func;
    }

}
