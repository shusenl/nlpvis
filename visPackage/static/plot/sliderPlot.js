class sliderPlot {
    constructor(svg, pos, size, label, range, value, format) {
        this.svg = svg.append("g")
            .attr("transform", "translate(" + pos[0] + "," + pos[1] + ")");
        this.pos = pos;
        this.size = size;
        this.label = label;
        this.range = range;
        this.value = value;
        this.format = format;
        this.draw();
    }

    mapValToWidth() {
        return (this.value - this.range[0]) /
            (this.range[1] - this.range[0]) *
            this.size[0]
    }

    bindUpdateCallback(callback) {
        this.callback = callback;
        callback(this.value);
    }

    draw() {
        //draw background quad
        let that = this;
        this.svg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.size[0])
            .attr("height", this.size[1])
            .attr("fill", "lightgrey")
            .on("click", function(d) {
                let coord = d3.mouse(this);
                that.value = that.range[0] + coord[0] / that.size[0] *
                    (that.range[1] - that.range[0]);
                // console.log(that.value);
                that.bar.attr("width", that.mapValToWidth());
                let val = d3.format(that.format)(that.value);
                that.value = Number(val);
                that.valText.text(val);
                // this.bar.attr("width", )
                that.callback(that.value);
            });

        //draw for ground quad
        this.bar = this.svg.append("rect")
            .attr("x", 0)
            .attr("x", 0)
            .attr("width", d => {
                return this.mapValToWidth();
            })
            .attr("height", this.size[1])
            .attr("fill", "grey")
            .style("pointer-events", "none");

        this.svg.append('text')
            .attr("x", -3)
            .attr("y", this.size[1] * 0.5)
            .text(this.label)
            .style("font-size", 13.0)
            .style("alignment-baseline", "middle")
            .style("text-anchor", "end")
            .style("pointer-events", "none");

        this.valText = this.svg.append('text')
            .attr("x", this.size[0] * 0.5)
            .attr("y", this.size[1] * 0.5)
            .text(this.value)
            .style("fill", "white")
            .style("font-size", 13.0)
            .style("alignment-baseline", "middle")
            .style("text-anchor", "middle")
            .style("pointer-events", "none");
    }
}
