class d3UIslider {
    constructor(svg, label, range, pos = [15, 15], size = [140, 50]) {
        this.svgContainer = svg;
        this._label = label;
        this._displayRange = range;
        this._range = [0, 100];
        this._pos = pos;
        this._size = size;
        this.id = 'd3UIslider-' + uuidv1();
        this.draw();
    }

    value() {
        if (this.svgContainer)
            return this.mapToDisplayRange(Number(this.svgContainer.select(
                    '#input' + this.id).node()
                .value));
    }

    mapToDisplayRange(value) {
        // console.log("mapToDisplayRange", value);
        return (this._displayRange[1] - this._displayRange[0]) / 100.0 *
            value + this._displayRange[0];
    }

    draw() {
        if (this.svgContainer.select("#fobject" + this.id).empty()) {
            this.fObject = this.svgContainer.append("foreignObject");
            this.fObject
                .attr('id', "fobject" + this.id)
                .attr("x", this._pos[0])
                .attr("y", this._pos[1])
                .attr("width", this._size[0])
                .attr("height", this._size[1]);
            this.htmlObject = this.fObject.append("xhtml") //this will cause issues in different browsers
                // .html(htmlSlider)

            this.htmlObject.append("label")
                .attr("id", "label" + this.id)
                .text(this._label)
                .style("font-size", "15px");

            this.htmlObject.append("input")
                .attr("id", "input" + this.id)
                .attr("type", "range")
                // .attr("width", (this._size[0] - 2) + "px")
                .attr("min", this._range[0])
                .attr("max", this._range[1])
                .attr("value", 20);


            var format = d3.format(".3n");
            var slider = this.svgContainer.select('#input' + this.id);
            var label = this.svgContainer.select('#label' + this.id);
            label.text(this._label + ": " + format(this.mapToDisplayRange(
                Number(slider.node().value))));

            slider.on("change", d => {
                if (this.updateCallback)
                    this.updateCallback();
            });

            slider.on("input", d => {
                // console.log(slider.node());
                label.text(this._label + ": " + format(this.mapToDisplayRange(
                    Number(slider.node().value))));
            });
        } else {
            // console.log("update pos - slider");
            this.fObject
                .attr("x", this._pos[0])
                .attr("y", this._pos[1]);
        }
    }

    pos(pos) {
        this._pos = pos;
        this.draw();
    }

    callback(func) {
        this.updateCallback = func;
    }
}
