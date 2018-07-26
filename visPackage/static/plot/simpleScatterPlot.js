class simpleScatterPlot {
    constructor(svg, pos, size, axisX = true, axisY = true) {
        this.svg = svg.append("g");
        this.update(pos, size);
        this.axisXflag = axisX;
        this.axisYflag = axisY;

        /// small button
    }

    draw() {
        if (this.data) {
            var data = this.data.map(this.accessor);
            this.svg.selectAll("*").remove();
            var xScale = d3.scaleLinear()
                .range([this.pos[0], this.pos[0] + this.size[0]]);

            var yScale = d3.scaleLinear()
                .range([this.pos[1] + this.size[1], this.pos[1]]);

            xScale.domain(d3.extent(data, d => d[0])).nice();
            yScale.domain(d3.extent(data, d => d[1])).nice();

            var r = 5;
            var points = this.svg.selectAll('.point')
                .data(data)
                .enter().append('circle')
                .attr('class', 'point')
                .attr('cx', function(d) {
                    return xScale(d[0]);
                })
                .attr('cy', function(d) {
                    return yScale(d[1]);
                })
                .attr('r', r)
                .style("opacity", 0.5)
                .style('fill', "lightgrey")
                .on("click", (d, i) => {
                    this.callback(this.data[i]);
                })
                .on("mouseover", function(d) {
                    // console.log("mouseOver");
                    d3.select(this).style("fill", "grey");
                    // d3.select(this).attr("r", 2*r);
                    //draw tooltip
                })
                .on("mouseout", function(d) {
                    d3.select(this).style("fill", "lightgrey");
                });

            if (this.axisXflag) {
                this.svg.append("g")
                    .attr("transform", "translate(" + this.pos[0] + ",0)")
                    .call(d3.axisLeft(yScale).ticks(5));
                this.svg.append('text')
                    .attr('x', this.pos[0] + 7)
                    .attr('y', this.pos[1] + 10)
                    .text(this.names[1])
                    .style("pointer-events", "none");
            }

            if (this.axisYflag) {
                this.svg.append("g")
                    .attr("transform", "translate(0," + (this.pos[1] + this
                        .size[1]) + ")")
                    .call(d3.axisBottom(xScale).ticks(5))
                this.svg.append('text')
                    .attr('x', this.pos[0] + this.size[0])
                    .attr('y', this.pos[1] + this.size[1] - 5)
                    .attr('text-anchor', 'end')
                    .text(this.names[0])
                    .style("pointer-events", "none");
            }
        }
    }

    update(pos, size) {
        //adjust for axis
        this.pos = [pos[0] + 25, pos[1] + 3];
        this.size = [size[0] - 35, size[1] - 20];
        this.draw();
    }

    setData(data, names, accessor = undefined, value = []) {
        this.data = data;
        this.accessor = accessor;
        this.val = value;
        this.names = names;
        this.draw();
    }

    bindSelectionCallback(callback) {
        this.callback = callback;
    }
}
