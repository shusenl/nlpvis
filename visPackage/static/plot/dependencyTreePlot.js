/*

add dependencyTreePlot to an existing svg
*/

class dependencyTreePlot {

    //pos: The index position
    //width: The width of sentence
    //heigth: The height of sentence
    constructor(svg, orientation, sen, pos, dep_triples, width, height) {
        this.svg = svg.append('g');
        // this.svg = svg;
        this.orientation = orientation;
        this.pos = pos;
        this.sen = sen;
        this.dep_triples = dep_triples;
        this.width = width;
        this.height = height;

        //text box width and height
        this.text_box_width = Math.min(this.width * 0.1, 15);
        this.text_box_height = Math.min(this.height * 0.1, 10);

        this.collapseIndex = new Set();

        this.highlight_indexs = [];

        this.filter(); //init the display index
        // console.log(dep_triples);
        this.draw();
    }

    clear() {
        this.svg.selectAll("*").remove();
    }

    getDepTreeData() {
        return this.dep_triples;
    }

    setCollapseHandle(func) {
        this.callback = func;
    }

    getCurrentMask() {
        sentenceMask = [];
        for (let i = 0; i < this.sen.length; i++) {
            if (this.display_index.includes(i)) {
                sentenceMask[i] = 1;
            } else {
                sentenceMask[i] = 0;
            }
        }
        return sentenceMask;
    }

    onHandleCollapse() {

        this.sentenceMask = [];
        for (let i = 0; i < this.sen.length; i++) {
            if (this.display_index.includes(i)) {
                this.sentenceMask[i] = 1;
            } else {
                this.sentenceMask[i] = 0;
            }
        }

        this.callback(this.sentenceMask); //[1,0,0,1]
    }

    //i: index of word in sentence
    collapse(i) {
        // this.display_index.indexOf(d[0])
        //if (this.display_index[i] !== i) {
        //correct index, when the not all words are displayed
        //    i = this.display_index[i];
        //}

        if (this.collapseIndex.has(i)) {
            this.collapseIndex.delete(i);
        } else {
            this.collapseIndex.add(i);
        }

        this.filter();

        //this.draw();
        //callback called, this will trigger a redraw
        this.onHandleCollapse();
        //this.draw();
    }

    //i: index of word in sentence
    highlight(i) {

        this.highlight_indexs = [];
        if (i != -1)
            this.highlight_indexs = this.getChild(i);

        this.draw();
    }

    //support function for collapse: filter
    //this.display_index is reset, shows which words are presented
    filter() {
        let childs = [];
        let display_index = [];
        // console.log("collapseIndex:", this.collapseIndex);

        this.collapseIndex.forEach(d => {
            let collapseChildren = this.getChild(d);
            // console.log(d, collapseChildren);
            childs = childs.concat(collapseChildren);
        });

        let childs_set = new Set(childs);
        // console.log("children set: ", childs_set);
        for (let i = 0; i < this.sen.length; i++) {
            if (!childs_set.has(i)) {
                display_index.push(i);
            }
        }
        this.display_index = display_index;
        // console.log(display_index);
    }

    //support function for collapse: get child index
    getChild(index) {
        // console.log(index, "------ deps:", deps);
        let childs = [];
        let filter = new Set();

        //filter hold the source / potential source of the arrow
        filter.add(index);

        //loop through the dependency until there is not new node
        //add to the filter.
        let l = 0;
        let depth = 0;
        do {
            l = filter.size;
            for (let i = 0; i < this.dep_triples.length; i++) {
                if (filter.has(this.dep_triples[i][0]) && !(filter.has(this
                        .dep_triples[i][2]))) {
                    filter.add(this.dep_triples[i][2]);
                    childs.push(this.dep_triples[i][2]);
                }
            }
        } while (filter.size != l);

        return childs;
    }

    updatePos(pos) {
        this.pos = pos;
        this.draw();
    }

    updateSize(width, height) {
        this.width = width;
        this.height = height;

        //text box width and height
        this.text_box_width = Math.min(this.width * 0.1, 15);
        this.text_box_height = Math.min(this.height * 0.1, 10);

        this.draw();
    }

    //draw the dependency tree over sentence
    draw() {

        //clean
        //this.svg.selectAll('.label,.arc, defs,g').remove();
        this.clear();

        //arrow
        // let arrowid = uuidv1()
        let arrowid = "depArrow";
        this.svg
            .append("defs")
            .append("marker")
            .attr("id", "depArrow")
            .attr('viewBox', '0 0 10 10')
            .attr("refX", 1)
            .attr("refY", 5)
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M 0 0 L 10 5 L 0 10 z")
            .style('fill', 'steelblue');

        //line function
        let lineFunction = d3.line()
            .x(function(d) {
                return d.x;
            })
            .y(function(d) {
                return d.y;
            })
            .curve(d3.curveLinear);

        //path
        let path_loc = this.pathLocation();
        this.svg.selectAll('.dep_tree_dep_path').data(path_loc)
            .enter()
            .append('path')
            .attr("class", "arc")
            .attr('d', function(d) {
                return lineFunction(d.data);
            })
            .attr("fill", "none")
            .attr("stroke", (d) => {
                return d.highlight ? 'orange' : "gray";
            })
            .attr("stroke-opacity", (d) => {
                return d.highlight ? 1 : 0.5;
            })
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
            .attr("stroke-width", 1.5)
            .style("stroke-dasharray", "4,4")
            .style("marker-end", "url(#" + arrowid + ")");

        //component rect
        let text_loc = this.textLocation();

        //component text
        this.svg.selectAll('.dep_tree_rel_text').data(text_loc)
            .enter()
            .append('text')
            .attr("class", "label")
            .text(function(d) {
                return d.text;
            })
            .attr('x', function(d, i) {
                return d.x;
            })
            .attr('y', function(d, i) {
                return d.y;
            })
            .attr('font-weight', (d) => {
                return d.highlight ? 'bold' : 'normal';
            })
            .attr('font-size', (d, i) => {
                return Math.min(12, d['width'] * 0.3) + 'px';
            })
            .style('writing-mode', (d) => {
                return this.orientation == 'v-left' ? 'vertical-lr' :
                    'horizontal-tb';
            })
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .on('mouseover', function() {
                d3.select(this).attr('font-size', 12);
            })
            .on('mouseout', function() {
                d3.select(this).attr('font-size', function(d) {
                    return Math.min(12, d['width'] * 0.3) +
                        'px';
                });
            })
    }


    //TODO: pathDepth is using brute force algorithm. optimize it by using dynamic programming
    PathDepth(start_node, end_node) {
        let depth = 1;

        let b_left = Math.min(start_node, end_node),
            b_right = Math.max(start_node, end_node);

        for (let i = 0; i < this.dep_triples.length; i++) {
            let triples = this.dep_triples[i];

            let left = Math.min(triples[0], triples[2]),
                right = Math.max(triples[0], triples[2]);


            let condition1 = b_left <= left && right < b_right;
            let condition2 = b_left < left && right <= b_right;

            if (condition1 || condition2) {
                let temp = 1 + this.PathDepth(triples[0], triples[2]);
                if (temp > depth)
                    depth = temp;
            }
        }
        return depth;
    }

    //get text pos
    textLocation() {
        let data = [];
        for (let i = 0; i < this.dep_triples.length; i++) {
            let d = this.dep_triples[i];

            if (this.display_index.includes(d[0]) && this.display_index.includes(
                    d[2])) {
                let word1_loc = this.pos[this.display_index.indexOf(d[0])],
                    word2_loc = this.pos[this.display_index.indexOf(d[2])],
                    item = {
                        'text': d[1],
                        'highlight': this.highlight_indexs.includes(d[2])
                    },
                    depth = this.PathDepth(d[0], d[2]);
                //depth = this.nodeDepth(d[2]);


                if (this.orientation == 'h-top') {
                    item['x'] = (word1_loc.x + word2_loc.x) / 2;
                    item['y'] = word1_loc.y - depth * this.text_box_height -
                        this.text_box_height * 1.5;
                    item['width'] = Math.abs(word1_loc.x - word2_loc.x)
                } else if (this.orientation == 'h-bottom') {
                    item['x'] = (word1_loc.x + word2_loc.x) / 2;
                    item['y'] = word1_loc.y + depth * this.text_box_height +
                        this.text_box_height * 1.5;
                    item['width'] = Math.abs(word1_loc.x - word2_loc.x)
                } else if (this.orientation == 'v-left') {
                    item['x'] = word1_loc.x - depth * this.text_box_width -
                        this.text_box_width * 1.5;
                    item['y'] = (word1_loc.y + word2_loc.y) / 2;
                    item['width'] = Math.abs(word1_loc.y - word2_loc.y)
                }
                data.push(item)
            }
        }
        return data;
    }

    //get path pos
    pathLocation() {
        let data = [];

        for (let i = 0; i < this.dep_triples.length; i++) {
            let d = this.dep_triples[i];
            if (this.display_index.includes(d[0]) && this.display_index.includes(
                    d[2])) {
                let word1_loc = this.pos[this.display_index.indexOf(d[0])],
                    word2_loc = this.pos[this.display_index.indexOf(d[2])],
                    item = {
                        'data': [],
                        'highlight': this.highlight_indexs.includes(d[2])
                    },
                    depth = this.PathDepth(d[0], d[2]);

                if (this.orientation == 'h-top') {
                    //first point
                    item.data.push({
                        'x': word1_loc.x,
                        'y': word1_loc.y - this.text_box_height *
                            1.5
                    });

                    //second point
                    item.data.push({
                        'x': word1_loc.x * 5 / 6 + word2_loc.x * 1 /
                            6,
                        'y': word1_loc.y - depth *
                            this.text_box_height - this
                            .text_box_height * 1.5
                    });
                    //third point
                    item.data.push({
                        'x': word1_loc.x * 1 / 6 + word2_loc.x * 5 /
                            6,
                        'y': word1_loc.y - depth *
                            this.text_box_height - this
                            .text_box_height * 1.5
                    });
                    //fourth point
                    item.data.push({
                        'x': word2_loc.x,
                        'y': word2_loc.y - this.text_box_height *
                            1.5
                    });
                } else if (this.orientation == 'h-bottom') {
                    //first point
                    item.data.push({
                        'x': word1_loc.x,
                        'y': word1_loc.y + this.text_box_height *
                            1.5
                    });
                    //second point
                    item.data.push({
                        'x': word1_loc.x * 5 / 6 + word2_loc.x * 1 /
                            6,
                        'y': word1_loc.y + depth *
                            this.text_box_height + this
                            .text_box_height * 1.5
                    });
                    //third point
                    item.data.push({
                        'x': word1_loc.x * 1 / 6 + word2_loc.x * 5 /
                            6,
                        'y': word1_loc.y + depth *
                            this.text_box_height + this
                            .text_box_height * 1.5
                    });
                    //fourth point
                    item.data.push({
                        'x': word2_loc.x,
                        'y': word2_loc.y + this.text_box_height *
                            1.5
                    });
                } else if (this.orientation == 'v-left') {
                    //first point
                    item.data.push({
                        'x': word1_loc.x - this.text_box_width *
                            2,
                        'y': word1_loc.y
                    });
                    //second point
                    item.data.push({
                        'x': word1_loc.x - depth *
                            this.text_box_width - this.text_box_width *
                            1.5,
                        'y': word1_loc.y * 5 / 6 + word2_loc.y * 1 /
                            6
                    });
                    //third point
                    item.data.push({
                        'x': word1_loc.x - depth *
                            this.text_box_width - this.text_box_width *
                            1.5,
                        'y': word1_loc.y * 1 / 6 + word2_loc.y * 5 /
                            6
                    });
                    //fourth point
                    item.data.push({
                        'x': word2_loc.x - this.text_box_width *
                            2,
                        'y': word2_loc.y
                    });
                } else
                    throw Error('unknow orientation');

                data.push(item);
            }
        }
        return data;
    }
}
