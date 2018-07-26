class treeMapPlot {
    constructor(svg, pos, size, title) {
        this.title = title;
        this.svg = svg.append("g");
        this.pos = pos;
        this.size = size;
        this.topMargin = 18;
    }

    generateNestData(data, levelTag, rootName) {

        var nestData;
        if (levelTag.length === 1) {
            nestData = d3.nest().key(d => d[levelTag[0]]).entries(data);

        } else { //only generate tree with depth 2
            nestData = d3.nest().key(d => {
                return d[levelTag[0]];
            }).key(d => {
                return d[levelTag[1]];
            }).entries(data);
        }

        var data = {
            key: rootName,
            values: nestData
        }

        // console.log(data);
        return data;
    }

    setTitle(title) {
        this.title = title;
        this.draw();
    }

    setData(treeData, rootName) {
        //test data
        this.data = this.generateNestData(treeData, ["correctness",
            "predict"
        ], rootName);
        this.draw();
    }

    update(pos, size) {
        this.pos = pos;
        this.size = size;
        this.draw();
    }

    draw() {
        this.drawSimple();
    }

    drawSimple() {
        this.svg.selectAll("*").remove();

        if (this.title) {
            this.svg.append("text")
                .text(this.title)
                .attr("x", this.pos[0] + 0.5 * this.size[0])
                .attr("y", this.pos[1] + 13)
                .attr("fill", "grey")
                .attr("text-anchor", "middle");
        }

        var width = this.size[0];
        var height = this.size[1];
        var pos = this.pos;

        var data = this.data;
        var svg = this.svg;
        const colormap = d3.scaleOrdinal().domain([
            "entailment-entailment",
            "neutral-neutral",
            "contradiction-contradiction",

            "entailment-contradiction",
            "contradiction-entailment",
            "neutral-entailment",
            "neutral-contradiction",
            "entailment-neutral",
            "contradiction-neutral",

        ]).range([
            // "#00B31A",
            // "#59D966",
            // "#B3FFB3",

            "#59D966",
            "#59D966",
            "#59D966",

            // "#FF5C05",
            // "#FF7726",
            // "#FF9248",
            // "#FFAA69",
            // "#FFC08B",
            // "#FFD4AD"

            "#FF9248",
            "#FF9248",
            "#FF9248",
            "#FF9248",
            "#FF9248",
            "#FF9248"
        ]);
        const treemap = d3.treemap().size([width, height]);

        // d3.json("flare.json", function(error, data) {
        //     if (error) throw error;

        const root = d3.hierarchy(data, d => d.values)
            .sum(d => {
                return d.index ? 1 : 0;
            })
            .sort(function(a, b) {
                return a.index - b.index;
            });


        const tree = treemap(root);
        // console.log(tree.descendants());
        let nodes = tree.descendants().filter(d => d.depth <= 2);
        svg.selectAll(".node")
            .data(nodes).enter().each(d => {
                // console.log(d);
                var g = svg.append("g");
                g.append("rect")
                    .attr("class", "node")
                    .attr("x", pos[0] + d.x0)
                    .attr("y", pos[1] + this.topMargin + d.y0)
                    .attr("width", Math.max(0, d.x1 - d.x0 - 1))
                    .attr("height", Math.max(0, d.y1 - d.y0 - 1))
                    .attr("fill", colormap(d.data.key))
                    .attr("stroke", "white")
                    .on("click", _ => {
                        this.selectCell(d);
                    }).on("mouseover", function(_) {
                        d3.select(this).attr("fill", "grey");
                    })
                    .on("mouseout", function(_) {
                        d3.select(this).attr("fill", colormap(d.data
                            .key));
                    });
                g.append("text")
                    .attr("x", pos[0] + (d.x0 + d.x1) * 0.5)
                    .attr("y", pos[1] + this.topMargin + (d.y0 + d.y1) *
                        0.5 + 5)
                    .text(_ => {
                        var str = d.data.key.replace("-", "/");
                        str = str.replace(/neutral/g, "N");
                        str = str.replace(/entailment/g, "E");
                        str = str.replace(/contradiction/g, "C");
                        return str;
                    })
                    .style("font-size", 14)
                    .style("writing-mode", _ => {
                        if (d.y1 - d.y0 > 2 * (d.x1 - d.x0))
                            return "vertical-rl";
                        else
                            return "hortizontal-rl";
                    })
                    .style("text-anchor", "middle")
                    .style("pointer-events", "none");
            });
        // .text((d) => d.data.name);

    }

    selectCell(d) {
        // console.log(d);
        var cellData = [];
        for (var i = 0; i < d.children.length; i++) {
            cellData.push(d.children[i].data);
        }
        // console.log(cellData);
        this.selectionCallback(cellData);
    }

    bindSelectionCallback(callback) {
        this.selectionCallback = callback;
    }

    //////////////////

    drawComplex() {
        var formatNumber = d3.format("d");

        var svg = this.svg;
        var data = this.data

        // var width = +this.svg.attr("width");
        // var height = +this.svg.attr("height");
        var width = 400;
        var height = 400;

        // var color = d3.scale.category20c();
        var color = d3.scaleOrdinal(d3.schemeCategory20c);

        var x = d3.scaleLinear()
            .domain([0, width])
            .range([0, width]);

        var y = d3.scaleLinear()
            .domain([0, height])
            .range([0, height]);

        const root = d3.hierarchy(data, d => d.values)
            .sum(d => d.value)
            .sort(function(a, b) {
                return a.value - b.value;
            });

        // root.children((d, depth) => {
        //     return depth ? null : d._children;
        // })

        var treemap = d3.treemap(root)
            .size([width, height])
            .round(false);

        // treemap(root.sum(function(d) {
        //         return d.value;
        //     }).sort(function(a, b) {
        //         return a.value - b.value;
        //     }))
        // .ratio(height / width * 0.5 * (1 + Math.sqrt(5)))
        // .round(false);

        // var svg = d3.select("#chart").append("svg")
        //     .attr("width", width + margin.left + margin.right)
        //     .attr("height", height + margin.bottom + margin.top)
        //     .style("margin-left", -margin.left + "px")
        //     .style("margin.right", -margin.right + "px")
        //     .append("g")
        //     .attr("transform", "translate(" + margin.left + "," + margin.top +
        //         ")")
        //     .style("shape-rendering", "crispEdges");

        var grandparent = svg.append("g")
            .attr("class", "grandparent");

        grandparent.append("rect")
            .attr("y", 0)
            .attr("width", width)
            .attr("height", 0);

        grandparent.append("text")
            .attr("x", 6)
            .attr("y", 6)
            .attr("dy", ".75em");


        // console.log(root);
        initialize(root);
        accumulate(root);
        layout(root);
        console.log(root);
        display(root);

        function initialize(root) {
            root.x = root.y = 0;
            root.dx = width;
            root.dy = height;
            root.depth = 0;
        }

        // Aggregate the values for internal nodes. This is normally done by the
        // treemap layout, but not here because of our custom implementation.
        // We also take a snapshot of the original children (_children) to avoid
        // the children being overwritten when when layout is computed.
        function accumulate(d) {
            return (d._children = d.values) ? d.value = d.values.reduce(
                function(p,
                    v) {
                    return p + accumulate(v);
                }, 0) : d.value;
        }

        // Compute the treemap layout recursively such that each group of siblings
        // uses the same size (1×1) rather than the dimensions of the parent cell.
        // This optimizes the layout for the current zoom state. Note that a wrapper
        // object is created for the parent node for each group of siblings so that
        // the parent’s dimensions are not discarded as we recurse. Since each group
        // of sibling was laid out in 1×1, we must rescale to fit using absolute
        // coordinates. This lets us use a viewport to zoom.
        function layout(d) {
            if (d._children) {
                treemap.nodes({
                    _children: d._children
                });
                d._children.forEach(function(c) {
                    c.x = d.x + c.x * d.dx;
                    c.y = d.y + c.y * d.dy;
                    c.dx *= d.dx;
                    c.dy *= d.dy;
                    c.parent = d;
                    layout(c);
                });
            }
        }

        function display(d) {
            grandparent
                .datum(d.parent)
                .on("click", transition)
                .select("text")
                .text(name(d));

            var g1 = svg.insert("g", ".grandparent")
                .datum(d)
                .attr("class", "depth");

            console.log(d._children);
            var g = g1.selectAll("g")
                .data(d._children)
                .enter().append("g");

            g.filter(function(d) {
                    return d._children;
                })
                .classed("children", true)
                .on("click", transition);

            var children = g.selectAll(".child")
                .data(function(d) {
                    return d._children || [d];
                })
                .enter().append("g");

            children.append("rect")
                .attr("class", "child")
                .call(rect)
                .append("title")
                .text(function(d) {
                    return d.key + " (" + formatNumber(d.value) + ")";
                });
            children.append("text")
                .attr("class", "ctext")
                .text(function(d) {
                    return d.key;
                })
                .call(text2);

            g.append("rect")
                .attr("class", "parent")
                .call(rect);

            var t = g.append("text")
                .attr("class", "ptext")
                .attr("dy", ".75em")

            t.append("tspan")
                .text(function(d) {
                    return d.key;
                });
            t.append("tspan")
                .attr("dy", "1.0em")
                .text(function(d) {
                    return formatNumber(d.value);
                });
            t.call(text);

            g.selectAll("rect")
                .style("fill", function(d) {
                    return color(d.key);
                });

            function transition(d) {
                if (transitioning || !d) return;
                transitioning = true;

                var g2 = display(d),
                    t1 = g1.transition().duration(750),
                    t2 = g2.transition().duration(750);

                // Update the domain only after entering new elements.
                x.domain([d.x, d.x + d.dx]);
                y.domain([d.y, d.y + d.dy]);

                // Enable anti-aliasing during the transition.
                svg.style("shape-rendering", null);

                // Draw child nodes on top of parent nodes.
                svg.selectAll(".depth").sort(function(a, b) {
                    return a.depth - b.depth;
                });

                // Fade-in entering text.
                g2.selectAll("text").style("fill-opacity", 0);

                // Transition to the new view.
                t1.selectAll(".ptext").call(text).style("fill-opacity", 0);
                t1.selectAll(".ctext").call(text2).style("fill-opacity", 0);
                t2.selectAll(".ptext").call(text).style("fill-opacity", 1);
                t2.selectAll(".ctext").call(text2).style("fill-opacity", 1);
                t1.selectAll("rect").call(rect);
                t2.selectAll("rect").call(rect);

                // Remove the old node when the transition is finished.
                t1.remove().each("end", function() {
                    svg.style("shape-rendering", "crispEdges");
                    transitioning = false;
                });
            }

            return g;
        }

        function text(text) {
            text.selectAll("tspan")
                .attr("x", function(d) {
                    return x(d.x) + 6;
                })
            text.attr("x", function(d) {
                    return x(d.x) + 6;
                })
                .attr("y", function(d) {
                    return y(d.y) + 6;
                })
                .style("opacity", function(d) {
                    return this.getComputedTextLength() < x(d.x + d.dx) -
                        x(d.x) ?
                        1 : 0;
                });
        }

        function text2(text) {
            text.attr("x", function(d) {
                    return x(d.x + d.dx) - this.getComputedTextLength() -
                        6;
                })
                .attr("y", function(d) {
                    return y(d.y + d.dy) - 6;
                })
                .style("opacity", function(d) {
                    return this.getComputedTextLength() < x(d.x + d.dx) -
                        x(d.x) ?
                        1 : 0;
                });
        }

        function rect(rect) {
            rect.attr("x", function(d) {
                    return x(d.x);
                })
                .attr("y", function(d) {
                    return y(d.y);
                })
                .attr("width", function(d) {
                    return x(d.x + d.dx) - x(d.x);
                })
                .attr("height", function(d) {
                    return y(d.y + d.dy) - y(d.y);
                });
        }

        function name(d) {
            return d.parent ? name(d.parent) + " / " + d.key + " (" +
                formatNumber(
                    d.value) + ")" : d.key + " (" + formatNumber(d.value) +
                ")";
        }
    }
}
