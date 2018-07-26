class pixelBar {
    constructor(svg, pos, size, att, words, ratio, colormap, metaInfo, active =
        true) {
        this.isActive = active;
        this.svg = svg.append("g");
        this.pos = pos;
        this.size = size;
        this.colormap = colormap;
        this.attData = att;
        this.words = words;
        this.metaInfo = metaInfo;

        this.selectionFlag = false;
        //default colormap
    }

    setAttData(values) {
        this.attData = values;
    }

    setColorMap(map) {
        this.colormap = map;
    }

    //emphasis the higher values
    setEmphasisRatio(ratio) {
        this.ratio = ratio;
        this.draw();
    }

    draw() {
        if (this.isActive) {
            this.rect = this.svg.append("rect")
                .attr("class", "senBlock")
                .attr("x", this.pos[0])
                .attr("y", this.pos[1])
                .attr("width", this.size[0])
                .attr("height", this.size[1])
                .attr("fill", "white")
                .attr("stroke", "lightgrey")
                // .attr("stroke-width", 2)
                .on("mouseover", function(d) {
                    // d3.selectAll(".senBlock").attr("stroke", "grey");
                    // d3.select(this).attr("stroke", "lightblue");
                    d3.select(this).attr("stroke-width", 4);
                })
                .on("mouseout", d => {
                    // d3.selectAll(".senBlock").attr("stroke", "grey");
                    d3.selectAll(".senBlock").attr("stroke-width", 2);
                })
                // .on("mouseover", function(d) {
                //     this.callback(this.word, this.attData);
                // })
                // .on("mouseout", function(d){
                //     this.callback(this.word, this.attData);
                // })
                .on("click", d => {
                    // console.log(this.rect.attr("stroke"));
                    if (this.rect.attr("stroke") === "grey") {
                        d3.selectAll(".senBlock").attr("stroke",
                            "lightgrey");
                        d3.selectAll(".senBlock").attr("opacity", 1.0);
                        d3.selectAll(".cell").attr("opacity", 1.0);
                        this.callback();
                    } else {
                        //set de-emphasis on all other block
                        d3.selectAll(".senBlock").attr("stroke",
                            "lightgrey");
                        d3.selectAll(".senBlock").attr("opacity", 0.5);
                        d3.selectAll(".cell").attr("opacity", 0.5);
                        //set highlight color
                        this.rect.attr("stroke", "grey");
                        this.rect.attr("opacity", 1.0);
                        this.svg.selectAll(".cell").attr("opacity", 1.0);
                        this.callback(this.words, this.attData, [this.pos[
                                0],
                            this.pos[1] + this.size[1]
                        ], [this.pos[0] + this.size[0], this.pos[
                                1] +
                            this.size[1]
                        ], this.metaInfo);
                    }
                    // if (this.rect.attr("opacity") > 0.9) {
                    //     d3.selectAll(".senBlock").attr("opacity", 1.0);
                    //     d3.selectAll(".cell").attr("opacity", 0.5);
                    //     this.rect.attr("opacity", 0.5);
                    //
                    //     this.svg.selectAll(".cell").attr("opacity", 1.0);
                    //     this.callback(this.words, this.attData);
                    // } else {
                    //     d3.selectAll(".senBlock").attr("opacity", 1.0);
                    //     d3.selectAll(".cell").attr("opacity", 1.0);
                    //     this.callback();
                    // }
                });
        } else {
            this.svg.append("rect")
                // .attr("class", "senBlock")
                .attr("x", this.pos[0])
                .attr("y", this.pos[1])
                .attr("width", this.size[0])
                .attr("height", this.size[1])
                .attr("fill", "white")
                .attr("stroke", "lightgrey")
        }
        //ratio adjust

        if (this.attData) {
            // console.log(this.attData);
            //readjust bar size
            this.cellData = [];
            let unit = this.size[0] / this.attData.length;
            let unitSum = 0;
            for (var i = 0; i < this.attData.length; i++) {
                this.cellData.push([this.pos[0] + i * unit, unit]);
            }

            for (var i = 0; i < this.cellData.length; i++) {
                if (this.isActive) {
                    this.svg.append("rect")
                        .attr("class", "cell")
                        .attr("x", this.cellData[i][0])
                        .attr("y", this.pos[1])
                        .attr("width", this.cellData[i][1])
                        .attr("height", this.size[1])
                        .attr("fill", this.colormap(this.attData[i]))
                        .attr("pointer-events", "none");
                } else {
                    this.svg.append("rect")
                        // .attr("class", "cell")
                        .attr("x", this.cellData[i][0])
                        .attr("y", this.pos[1])
                        .attr("width", this.cellData[i][1])
                        .attr("height", this.size[1])
                        .attr("fill", this.colormap(this.attData[i]))
                        .attr("pointer-events", "none");

                }
            }
        }
    }

    bindShowSentenceCallback(callback) {
        this.callback = callback;
    }
}
