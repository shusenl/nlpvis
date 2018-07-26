class d3UIcolorMap {
    constructor(svg, id, range = [0, 1], pos = [15, 15], size = [150, 20],
        ticks = 4, colorMapIndex = 0) {
        // this.svgTag = "#" + svgTag;
        this.svgContainer = svg;
        this._pos = pos;
        this._size = size;
        this._ticks = ticks;
        this._range = range;
        this._id = id;

        /////// pre-defined colormaps ///////
        this.colorMap = [];
        this.colorMap.push({
            'name': 'yellow-green-blue',
            // 'data': ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8",
            //     "#253494"
            // ]
            'data': ["#253494", "#2c7fb8", "#41b6c4", "#a1dab4",
                "#ffffcc"
            ]
        });
        this.colorMap.push({
            'name': 'hot',
            'data': ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20",
                "#bd0026"
            ]
        });
        this.colorMap.push({
            'name': 'diverging-brown-green',
            'data': ["#a6611a", "#dfc27d", "#f5f5f5", "#80cdc1",
                "#018571"
            ]
        });
        this.colorMap.push({
            'name': 'diverging-red-blue',
            'data': ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de",
                "#0571b0"
            ]
        });
        this.colorMap.push({
            'name': 'diverging-spectral',
            'data': ["#d53e4f", "#fc8d59", "#fee08b", "#ffffbf",
                "#e6f598", "#99d594", "#3288bd"
            ]
        });
        this.cmIndex = colorMapIndex;
        this.currentColormap = this.colorMap[this.cmIndex];

        ///// first draw /////
        this.draw();
    }

    draw() {
        this.svgContainer.select('#' + this._id).remove();
        this.svg = this.svgContainer.append("g").attr("id", this._id)
            .attr("transform", "translate(" + this._pos[0] + "," + this._pos[
                    1] +
                ")");
        this.defs = this.svg.append("defs");
        this._setupGradient();


        //add colormap selector
        this.cSelector = new d3UIitemSelector(this.svg,
            this.colorMap.map(d => d.name), this.cmIndex, [this._size[0] +
                1, 0
            ], [
                20, 20
            ], "20px");
        this.cSelector.callback(this.colormap.bind(this));

        //draw colorbar
        var colorbar = this.svg.append("rect")
            .attr('id', this._id + 'colorbar')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', this._size[0])
            .attr('height', this._size[1])
            .style("stroke", "grey")
            .style("stroke-width", 1.0)
            // .on("click", this.drawColorMapPicker.bind(this));
            // .on("click", this.cSelector.triggerDropdown.bind(this));

        this.svg.append("g").attr('id', this._id + "colorPicker");

        //set default colormap
        this.colormap(this.cmIndex);

        //draw axis
        this.drawAxis();

    }

    hide() {

    }

    show() {

    }

    resize() {

    }

    pos(pos) {
        this._pos = pos;
        this.svg = this.svgContainer.select('#' + this._id)
            .attr("transform", "translate(" + this._pos[0] + "," + this._pos[
                    1] +
                ")");
        this.cSelector.pos([this._size[0] + 1, 0]);
        // this.draw();
    }

    drawAxis() {
        if (this._range) {
            var scale = d3.scaleLinear().domain(this._range).range([0,
                this._size[
                    0]
            ]);
            var axis = d3.axisTop(scale).ticks(this._ticks).tickFormat(d3.format(
                ".2f"));
            var g = this.svg.append('g')
                .attr("transform", "translate(" + 0 + "," + 5 + ")")
                .attr("class", "axis")
                .call(axis);
            g.selectAll('text').style("font-size", "14px");
            g.selectAll("path")
                .style("fill", "none")
                .style("stroke", "none");
        }
    }

    svg() {
        return this.svg;
    }

    setReverseFlag(flag) {
        this.reverseFlag = flag;
        this.draw();
    }

    colormap(index) {
        this.cmIndex = index;
        this.currentColormap = this.colorMap[this.cmIndex];
        this._generateColorlookup();
        this.svg.select('#' + this._id + 'colorbar')
            .style("fill", "url(#" + this._id + "lg-cm-" + this.colorMap[
                this.cmIndex].name + ")");

        if (this.updateCallback) {
            // console.log('d3UIcolorMap:colormap=> colormap is updated\n');
            this.updateCallback(this);
        }
    }

    range(range) {
        this._range = range;
        // this._generateColorlookup();
        this.draw();
    }

    callback(func) {
        this.updateCallback = func;
    }

    lookup(value) {
        // console.log(this.valueScale.domain());
        return this.valueScale(value);
    }

    ticks(ticks) {
        if (ticks)
            this.tickes = ticks;
        else
            return this._ticks;
    }

    ////////// internal helper function //////////////
    _setupGradient() {

        var gradients = this.defs
            .selectAll("linearGradient")
            .data(this.colorMap)
            .enter().append("linearGradient")
            .attr("id", d => this._id + "lg-cm-" + d.name)
            .each(function(d, j) {
                var colormaplen = d.data.length;
                // console.log(d, j);
                d3.select(this)
                    .selectAll("stop")
                    .data(d.data)
                    .enter().append("stop")
                    .attr("offset", (d, i) => i / (colormaplen - 1))
                    .attr("stop-color", d => d);
            });
    }

    _generateColorlookup() {
        var cmrange = [];
        var cmlen = this.currentColormap.data.length;
        for (var i = 0; i < cmlen; i++) {
            cmrange.push(this._range[0] + (i / (cmlen - 1)) * (this
                ._range[
                    1] - this._range[0]));
        }
        this.valueScale = d3.scaleLinear()
            .domain(cmrange)
            .range(this.currentColormap.data)
            // .interpolate(d3.interpolateHsl);
            .interpolate(d3.interpolateRgb);
    }

}

/////// independent function for generating colormap ///////

function generateColormap(valueRange, colorList) {
    var cmrange = [];
    var cmlen = colorList.length;
    for (var i = 0; i < cmlen; i++) {
        cmrange.push(valueRange[0] + (i / (cmlen - 1)) * (valueRange[
            1] - valueRange[0]));
    }
    var colorMap = d3.scaleLinear()
        .domain(cmrange)
        .range(colorList)
        // .interpolate(d3.interpolateHsl);
        .interpolate(d3.interpolateRgb);

    return colorMap;
}
