/*
Handle multiple histogram
*/

class histoPlot {
    constructor(svg, pos, size, axisX = false, axisY = false,
        hist = [], sample = [], mode = "sample") {
        this.svg = svg.append("g");

        this.mode = mode;
        this.pos = pos;
        this.size = size;
        this.axisXflag = axisX;
        this.axisYflag = axisY;
        this.hist = hist;
        this.sample = sample;

        this.leftOffset = 10;
        this.bottomOffset = 18;

        this.draw();
    }

    setTitle(title) {
        this.title = title;
        this.draw();
    }

    setSample(sample, accessor) {
        this.accessor = accessor;
        this.sample = sample;
        this.mode = "sample";
        this.draw();
    }

    setHisto(histoList) {
        this.hist = histoList;
        this.mode = "hist";
        this.draw();
    }

    update(pos, size) {
        this.pos = pos;
        this.size = size;
        this.draw();
    }

    draw() {
        this.svg.selectAll("*").remove();
        var pos = this.pos;
        var width = this.size[0] - (this.axisYflag ? this.leftOffset : 0);
        var height = this.size[1] - (this.axisXflag ? this.bottomOffset : 0);
        // console.log(width, height);
        var binNum = 10;
        var x, y, bins;

        if (this.title) {
            this.svg.append("text")
                .text(this.title)
                .attr("x", this.pos[0] + 0.5 * this.size[0])
                .attr("y", this.pos[1] + 20)
                .attr("fill", "grey")
                .attr("text-anchor", "middle");
        }

        if (this.mode === "sample") {
            var samples = this.sample;
            if (samples.length === 0)
                return;
            x = d3.scaleLinear()
                .domain(d3.extent(samples, this.accessor))
                .range([this.pos[0], this.pos[0] + width])
                .nice(binNum);
            var histogram = d3.histogram()
                .domain(x.domain())
                .thresholds(binNum)
                .value(this.accessor);
            bins = histogram(samples);
            y = d3.scaleLinear()
                .domain([0, d3.max(bins, d => d.length)])
                .range([height + pos[1], pos[1]]);
            // console.log(bins);
            var bar = this.svg.selectAll(".bar")
                .data(bins)
                .enter().append("rect")
                .attr("class", "bar")
                // .attr("x", 1)
                .attr("fill", "lightgrey")
                .attr("transform", function(d) {
                    return "translate(" + x(d.x0) + "," + y(d.length) +
                        ")";
                })
                .attr("width", function(d) {
                    return x(d.x1) - x(d.x0) - 1;
                })
                .attr("height", function(d) {
                    return pos[1] + height - y(d.length);
                })
                .on("click", (d, i) => {
                    this.callback(d);
                })
                .on("mouseover", function(d) {
                    d3.select(this).style("fill", "grey");
                })
                .on("mouseout", function(d) {
                    d3.select(this).style("fill", "lightgrey");
                });
        } else if (this.mode === "hist") {
            //directly provide the histogram bin size
            let barWidth = width / this.hist[0].length - 1;
            x = d3.scaleLinear()
                .domain(d3.extent(this.hist[1]))
                .range([this.pos[0], this.pos[0] + width]);
            y = d3.scaleLinear()
                .domain([0, d3.max(this.hist[0])])
                .range([height + pos[1], pos[1]]);
            var bar = this.svg.selectAll(".bar")
                .data(this.hist[0])
                .enter().append("rect")
                .attr("class", "bar")
                .attr("fill", "lightgrey")
                .attr("x", (d, i) => {
                    return x(this.hist[1][i]);
                })
                .attr("width", barWidth)
                .attr("y", function(d) {
                    return y(d);
                })
                .attr("height", function(d) {
                    return pos[1] + height - y(d);
                }).on("click", (d, i) => {
                    this.callback(d);
                })
                .on("mouseover", function(d) {
                    d3.select(this).style("fill", "grey");
                })
                .on("mouseout", function(d) {
                    d3.select(this).style("fill", "lightgrey");
                });
        }


        // add the x Axis
        if (this.axisXflag) {
            this.svg.select("#xAxis").remove();
            this.svg.append("g")
                .attr("id", "xAxis")
                .attr("transform", "translate(0," + (pos[1] + height) +
                    ")")
                .call(d3.axisBottom(x).ticks(4));
        }
        // add the y Axis
        if (this.axisYflag) {
            this.svg.select("#yAxis").remove();
            this.svg.append("g")
                .attr("id", "yAxis")
                .attr("transform", "translate(" + this.pos[0] + ",0)")
                .call(d3.axisLeft(y).ticks(4));
        }
    }

    bindSelectionCallback(callback) {
        this.callback = callback;
    }
}
