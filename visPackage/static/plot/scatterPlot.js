///////////// scatterplot matrix d3 v4 version /////////////

class scatterPlot {
    constructor(svg, pos, size, renderMode = 'd3', plotType = 'single') {
        this.pos = pos;
        this.size = size;
        this.renderMode = renderMode; //'canvas' or 'd3'
        this.plotType = plotType; //'matrix' or 'single'
        this.plots = [];
        this.margin = {
            top: 50,
            right: 20,
            bottom: 55,
            left: 45
        };
        this.width = this.size[0];
        this.height = this.size[1];
        this.listOfIndex = [-1];
    }

    update(pos, size) {
        this.pos = pos;
        this.size = size;
        this.width = this.size[0];
        this.height = this.size[1];
    }

    setData(data, names, domain = [0, 1], values = [2]) {
        this.data = data;
        this.names = names;
        this.xIndex = domain[0];
        this.yIndex = domain[1];
        this.vIndex = values[0];
        this.draw();
    }

    setSeg(seg) {
        if (this._isValid()) {
            var c10 = d3.schemeCategory10();
            this.svg.selectAll("circle")
                .style("fill", (d, i) => c10(seg[i]));
            this.vSelector.setTempLabel("class");
        }
    }

    bindSelectionCallback(func) {
        this.selectionCallback = func;
    }

    bindSubselectionCallback(func) {
        this.subselectionCallback = func;
    }

    drawSPLOM() {

    }

    drawScalable() {

    }


    drawD3() {
        if (this.data) {
            // console.log(this.data);
            // this._updateWidthHeight();
            var zip = function(data) {
                let a = data[0];
                let b = data[1];
                let c = data[2];
                // console.log(a, b, c);
                return a.map((e, i) => [e, b[i], c[i]])
            }

            this.plotData = zip([this.data[this.xIndex],
                this.data[this.yIndex],
                this.data[this.vIndex]
            ]);

            this.x = d3.scaleLinear().domain(d3.extent(this.plotData, d =>
                d[0])).range(
                [0, this.width]);
            this.y = d3.scaleLinear().domain(d3.extent(this.plotData, d =>
                d[1])).range(
                [this.height, 0]);
            this.xAxis = d3.axisBottom(this.x).tickFormat(
                d3.format(".2f"));
            this.yAxis = d3.axisLeft(this.y).tickFormat(
                d3.format(".2f"));

            //cleanup
            d3.select(this._div).select("svg").remove();

            this.svg = d3.select(this._div).append("svg")
                .attr('width', this.pwidth)
                .attr('height', this.pheight)
                .append('g')
                .attr('transform', 'translate(' + this.margin.left + "," +
                    this
                    .margin.top + ")");

            //create background
            this.svg.append('rect')
                .attr('id', 'backgroundRect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', this.pwidth)
                .attr('height', this.pheight)
                .attr('fill', 'white');

            this.colormap = new d3UIcolorMap(this.svg,
                "scatterPlotColorBar" + this._divTag, d3.extent(this.plotData,
                    d =>
                    d[2]), this._cmOffset());
            this.xSelector =
                new d3UIitemSelector(this.svg, this.names, this.xIndex,
                    this._xsOffset());
            this.ySelector =
                new d3UIitemSelector(this.svg, this.names, this.yIndex,
                    this._ysOffset());
            this.vSelector =
                new d3UIitemSelector(this.svg, this.names, this.vIndex,
                    this._vsOffset());

            //add axes
            this.svg.append('g').attr('class', 'axis')
                .attr("id", "x-axis")
                .attr('transform', 'translate(0,' + this.height + ")").call(
                    this.xAxis);
            this.svg.append('g').attr('class', 'axis')
                .attr("id", "y-axis")
                .attr('class', 'axis').call(this.yAxis);

            //add points
            this.svg.selectAll(".scatterplotPoints")
                .data(this.plotData)
                .enter()
                .append('circle')
                .attr("r", 3.5)
                .attr('cx', d => this.x(d[0]))
                .attr('cy', d => this.y(d[1]))
                .attr('class', "scatterplotPoints")
                .style("fill", d => this.colormap.lookup(d[2]))
                .style('stroke', 'grey')
                .style('stroke-width', 0.5)
                .on("click", (d, i) => {
                    d3.event.stopPropagation();
                    this.selectionCallback(i);
                });

            var dragBehavior = d3.behavior.drag();
            var that = this;
            dragBehavior.on('dragstart', function() {
                // console.log(d);
                that.selectRectStart = d3.mouse(this);
                // d3.event.stopPropagation();
            });

            dragBehavior.on('drag', function() {
                that.selectRectEnd = d3.mouse(this);

                var x = Math.min(that.selectRectStart[0], that.selectRectEnd[
                    0]);
                var y = Math.min(that.selectRectStart[1], that.selectRectEnd[
                    1]);
                var width = Math.abs(that.selectRectStart[0] -
                    that.selectRectEnd[0]);
                var height = Math.abs(that.selectRectStart[1] -
                    that.selectRectEnd[1]);

                d3.select(that._div).select("svg")
                    .selectAll('.selectRect').remove();
                d3.select(that._div).select("svg")
                    .append('rect')
                    .attr('x', x)
                    .attr('y', y)
                    .attr('width', width)
                    .attr('height', height)
                    .attr('class', 'selectRect')
                    .style('fill', 'none')
                    .style('stroke', 'lightgrey')
                    .style('stroke-dasharray', '10,5')
                    .style('stroke-width', '3');

                //compute what point is in the rect
                that.listOfIndex = []
                that.svg.selectAll(".scatterplotPoints")
                    .each((d, i) => {
                        // console.log(d);z
                        if ( // inner circle inside selection frame
                            that.margin.left + that.x(d[0]) >=
                            x &&
                            that.margin.left + that.x(d[0]) <=
                            x + width &&
                            that.margin.top + that.y(d[1]) >= y &&
                            that.margin.top + that.y(d[1]) <= y +
                            height
                        )
                            that.listOfIndex.push(i);
                    });
                that.isDragging = true;

                //update highlight during drag movement
                that.updateHighlight(that.listOfIndex);

            });

            dragBehavior.on('dragend', d => {
                if (this.isDragging) {
                    d3.select(that._div).select("svg")
                        .selectAll('.selectRect').remove();
                    //update selection
                    this.subselectionCallback(this.listOfIndex);
                    this.isDragging = false;
                }
            });
            d3.select(this._div).select("svg").call(dragBehavior);


            //handle point deselection
            d3.select(this._div)
                .select("svg")
                .select('#backgroundRect')
                .on("click", d => {
                    this.listOfIndex = [];
                    //clear up selection
                    this.subselectionCallback(this.listOfIndex);
                    this.selectionCallback(-1);
                    console.log("background svg is clicked\n");
                });

            this.colormap.callback(this.updateColor.bind(this));
            this.xSelector.callback(this.updateAxisX.bind(this));
            this.ySelector.callback(this.updateAxisY.bind(this));
            this.vSelector.callback(this.updateValue.bind(this));

            this.colormap.draw();
        }
    }

    draw() {
        if (this.renderMode === "d3") {
            this.drawD3();
        } else if (this.renderMode === "Scalable") {
            this.drawScalable();
        }
    }

    resizeD3() {
        if (this._isValid()) {
            // this._updateWidthHeight();

            d3.select(this.svg.node().parentNode).style('width', this.pwidth)
                .style(
                    'height', this.pheight);
            this.svg.style('width', this.pwidth).style('height', this.pheight);
            // this.svg.select('g').attr('width', this.width).attr('height', this.height);
            this.colormap.pos(this._cmOffset());
            this.xSelector.pos(this._xsOffset());
            this.ySelector.pos(this._ysOffset());
            this.vSelector.pos(this._vsOffset());

            this.x.range([0, this.width]);
            this.y.range([this.height, 0]);

            this.updateAxisX(this.xIndex);
            this.updateAxisY(this.yIndex);
        }
    }

    resize() {
        if (this.renderMode === "d3") {
            this.resizeD3();
        } else if (this.renderMode === "Scalable") {
            this.resizeScalable();
        }
    }

    updatePos(px, py, method) {
        // console.log("scatterplots: update axis X", index);
        for (var i = 0; i < this.plotData.length; i++) {
            this.plotData[i][0] = px[i];
            this.plotData[i][1] = py[i];
        }
        this.x.domain(d3.extent(this.plotData, d => d[0]));
        this.y.domain(d3.extent(this.plotData, d => d[1]));

        this.yAxis.scale(this.y);
        this.svg.select("#y-axis").call(this.yAxis);

        this.xAxis.scale(this.x);
        this.svg.select("#x-axis")
            .attr('transform', 'translate(0,' + this.height + ")")
            .call(this.xAxis);

        this.svg.selectAll("circle")
            .attr('cx', d => this.x(d[0]))
            .attr('cy', d => this.y(d[1]));

        this.xSelector.setTempLabel(method + '_0');
        this.ySelector.setTempLabel(method + '_1');
    }

    updateAxisX(index) {
        if (this.xIndex !== index) {
            this.xIndex = index;
            // console.log("scatterplots: update axis X", index);
            for (var i = 0; i < this.plotData.length; i++) {
                this.plotData[i][0] = this.data[index][i];
            }
            this.x.domain(d3.extent(this.plotData, d => d[0]));
        }

        this.xAxis.scale(this.x);
        this.svg.select("#x-axis")
            .attr('transform', 'translate(0,' + this.height + ")")
            .call(this.xAxis);
        this.svg.selectAll("circle").attr('cx', d => this.x(d[0]));
    }

    updateAxisY(index) {
        if (this.yIndex !== index) {
            this.yIndex = index;
            // console.log("scatterplots: update axis Y:", index);
            for (var i = 0; i < this.plotData.length; i++) {
                this.plotData[i][1] = this.data[index][i];
            }
            this.y.domain(d3.extent(this.plotData, d => d[1]));
        }
        this.yAxis.scale(this.y);
        this.svg.select("#y-axis").call(this.yAxis);
        this.svg.selectAll("circle").attr('cy', d => this.y(d[1]));
    }

    updateValue(index) {
        // console.log("scatterplots: update value");
        for (var i = 0; i < this.plotData.length; i++) {
            this.plotData[i][2] = this.data[index][i];
        }
        this.colormap.range(d3.extent(this.plotData, d => d[2]));
        this.svg.selectAll("circle")
            .style("fill", d => this.colormap.lookup(d[2]));
    }

    updateColor(colormap) {
        this.svg.selectAll("circle")
            .style("fill", d => this.colormap.lookup(d[2]));
    }

    updateHighlight(indexList = []) {
        if (this._isValid()) {
            //have invalid index -1
            if (indexList.indexOf(-1) > -1)
                return;

            if (indexList.length === 0) {
                //clear highlight
                this.svg.selectAll("circle").style("fill-opacity", 1.0);
                this.svg.selectAll("circle").style("stroke", "grey");
            } else {
                // console.log(indexList);
                //set highlight
                this.svg.selectAll("circle").style("fill-opacity", (d, i) => {
                    if (indexList.indexOf(i) > -1) {
                        return 1.0;
                    } else {
                        return 0.03;
                    }
                }).style("stroke", (d, i) => {
                    if (indexList.indexOf(i) > -1) {
                        return "black";
                    } else {
                        return "grey";
                    }
                });
            }
        }
    }

    _cmOffset() {
        return [this.width - 285, -30];
    }
    _xsOffset() {
        return [this.width - 100, this.height + 22];
    }
    _ysOffset() {
        return [0, -30];
    }
    _vsOffset() {
        return [this.width - 105, -30];
    }
}
