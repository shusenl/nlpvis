/*

Matrix representation of attention

*/


class attentionMatrixComponent extends attentionComponent {

    constructor(uuid) {
        super(uuid);

        this.freeze_flag = {
            'index': -1,
            'flag': false
        };

        this.backgourndText = ['Premise', 'Hypothesis'];
    }

    draw() {
        this._updateWidthHeight();

        if (this.data["attention"] !== undefined && this.data["currentPair"]
            ["sentences"] !==
            undefined) {

            //init svg
            this.initSvg();

            //attention matrix
            let attMax = 1.0;
            let attMin = 0.0;
            if (this.comparisonFlag) {
                attMax = 1.0;
                attMin = -1.0;
                this.colorbar =
                    new d3UIcolorMap(this.svg, this.uuid, [attMin, attMax], [
                        10,
                        10
                    ], [this.width * 0.22, 24], 2, 3);
            } else {
                this.colorbar =
                    new d3UIcolorMap(this.svg, this.uuid, [attMin, attMax], [
                        10,
                        10
                    ], [this.width * 0.22, 24], 2);
            }
            ////////////////////add colormap //////////////////////

            this.colorbar.draw();
            this.colorbar.callback(this.updateMatrixColormap.bind(this));

            //data

            //location of words
            var pair = this.data["currentPair"]["sentences"];
            this.srcWords = this.sen2words(pair[0]);
            this.targWords = this.sen2words(pair[1]);

            this.aggregatedMatrix = Object.assign(this.normAttention);

            //for subMatrix
            if (this.data["selectionRange"]) {
                // console.log(this.data["selectionRange"]);
                this.srcWords = this.srcWords.slice(this.data[
                        "selectionRange"][0],
                    this.data["selectionRange"][1]);
                this.aggregatedMatrix = this.aggregatedMatrix.slice(
                    this.data["selectionRange"][0],
                    this.data["selectionRange"][1]);
            }

            this.attList = this.generateMatrixGeometry();

            this.computeWordPosition(this.srcWords, this.targWords);

            this.drawAttDisplayToggle(this.svg, [2, 50]);
            this.drawAttDataToggle(this.svg, [15, 80], 'horizontal');
            this.drawDepTree();

            //matrix
            this.rectw = (this.width * (3 / 4)) / this.targWords.length;
            this.recth = (this.height * (3 / 4)) / this.srcWords.length;


            let rects = this.svg.selectAll(
                    '.attentionComponent_matrix_rect')
                .data(this.attList)
                .enter()
                .append('rect')
                .attr('class', 'attentionComponent_matrix_rect')
                .attr('x', (d, i) => {
                    return d.x;
                })
                .attr('y', (d, i) => {
                    return d.y;
                })
                .attr('width', (d) => {
                    return d.width;
                })
                .attr('height', (d) => {
                    return d.height;
                })
                .style('stroke', 'black')
                .style('stroke-width', '1px')
                .style('fill', d => {
                    return this.colorbar.lookup(d.value);
                })
                .on('click', (d, i, nodes) => {

                    if (!this.freeze_flag.flag) {
                        this.freeze_flag.flag = true;
                        this.freeze_flag.index = i;
                        this.popSliderBar(d, i, nodes);
                    } else {
                        this.freeze_flag.flag = !(this.freeze_flag.index ==
                            i);
                    }
                })
                .on('mouseover', (d, i) => {

                    if (this.freeze_flag.flag) return;

                    let targWords = this.sen2words(this.data[
                            "currentPair"]["sentences"][1]),
                        col = i % targWords.length,
                        row = Math.floor(i / targWords.length);

                    this.highlight('highlight', row, col);
                    //update highlight
                    this.setData("highlight", [row, col]);
                })
                .on('mouseout', (d, i) => {
                    if (this.freeze_flag.flag) return;

                    let targWords = this.sen2words(this.data[
                            "currentPair"]["sentences"][1]),
                        col = i % targWords.length,
                        row = Math.floor(i / targWords.length);
                    this.highlight('clean', -1, -1);
                    //update highlight
                    this.setData("highlight", [-1, -1]);
                });

            this.rectCell = rects;
            /////////////////////// draw text /////////////////////////
            let texts = this.generateTextGeometry();
            //Draw targ text
            let targtext = this.svg.selectAll(
                    '.attentionComponent_targWords')
                .data(texts.targ)
                .enter()
                .append('text')
                .text(d => d.text)
                .attr('class', 'attentionComponent_targWords')
                .attr('x', (d, i) => d.x)
                .attr("y", (d, i) => d.y)
                .attr('display', d => d.display)
                .attr("transform", (d, i) => {
                    return "rotate(-45, " + d.x + ' , ' +
                        d.y + ')';
                })
                .attr('text-anchor', 'middle')
                .classed('attentionMatrixComponent_text_normal', true)
                .on('mouseover', (d, i, nodes) => {
                    // this.highlight(i);
                    if (this.targ_dep && !this.freeze_flag.flag)
                        this.targ_dep.highlight(i);
                })
                .on('mouseout', (d, i, nodes) => {
                    // this.highlight(-1);
                    if (this.targ_dep && !this.freeze_flag.flag)
                        this.targ_dep.highlight(-1);
                })
                .on('click', (d, i, nodes) => {
                    this.collapse(i, nodes, 'vertical');
                });
            this.targtext = targtext;

            //Draw src text
            let srctext = this.svg.selectAll('.attentionComponent_srcWords')
                .data(texts.src)
                .enter()
                .append('text')
                .text(d => d.text)
                .attr('class', 'attentionComponent_srcWords')
                .attr('x', (d, i) => d.x)
                .attr("y", (d, i) => d.y)
                .attr('display', d => d.display)
                .attr('text-anchor', 'middle')
                .classed('attentionMatrixComponent_text_normal', true)
                .on('mouseover', (d, i) => {
                    if (this.src_dep && !this.freeze_flag.flag)
                        this.src_dep.highlight(i);
                })
                .on('mouseout', (d, i) => {
                    if (this.src_dep && !this.freeze_flag.flag)
                        this.src_dep.highlight(-1);
                })
                .on('click', (d, i, nodes) => {
                    this.collapse(i, nodes, 'horizontal');
                });
            this.srctext = srctext;

            ////////////// rect mouse over event ///////////////
            //this.rectMouseEvent(rects, targtext, srctext);
            //draw text background only when depTree do not exist
            if (this.srcDepTreeData === undefined) {
                this.svg.selectAll(
                        '.attentionMatrixComponent_background_text')
                    .data([this.backgourndText[0], this.backgourndText[1]])
                    .enter()
                    .append('text')
                    .text(d => d)
                    .attr('x', (d, i) => {
                        return i == 0 ? this.width / 32 : this.width *
                            5 /
                            8;
                    })
                    .attr("y", (d, i) => {
                        return i == 0 ? this.height * 5 / 8 : this.width /
                            16;
                    })
                    .style('writing-mode', (d, i) => {
                        return i == 0 ? 'vertical-lr' : 'horizontal-tb';
                    })
                    .classed('attentionMatrixComponent_background_text',
                        true);
            }
        }
    }

    popSliderBar(d, i, nodes) {

        let width = Math.min(200, Math.max(this.width * 0.1, 80));
        let height = 8;
        let x = d.x;
        let y = d.y;
        let rectw = d.width;
        let recth = d.height;
        let circle_r = 8;
        let scale = d3.scaleLinear().domain([0, 1]).range([d.x - width / 2 +
            d.width / 2, d.x + width / 2 + d.width / 2
        ]);


        this.slider_bar_background = this.svg.append('rect').datum(d)
            .attr('x', (d) => {
                return d.x - width / 2 + d.width / 2 - 10;
            })
            .attr('y', (d) => {
                return d.y - d.height * 1.5;
            })
            .attr('width', width + 20)
            .attr('height', recth * 1.5)
            .attr('fill', 'white')
            .attr('rx', 5)
            .attr('ry', 5)
            .style('stroke', 'gray')
            .on('click', (d) => {
                this.slider_bar_circle.remove();
                this.slider_bar_background.remove();
                this.slider_bar.remove();
                this.slider_bar_axis.remove();
                this.freeze_flag.flag = false;
            });

        this.slider_bar = this.svg.append('rect').datum(d)
            .attr('x', (d) => {
                return d.x - width / 2 + d.width / 2;
            })
            .attr('y', (d) => {
                return d.y - d.height;
            })
            .attr('width', width)
            .attr('height', height)
            .style('fill', '#cfcbdb')
            .attr('rx', 2)
            .attr('ry', 2);

        this.slider_bar_axis = this.svg.append('g')
            .attr('transform', 'translate(0,' + (d.y - d.height + 4) + ')')
            .call(d3.axisBottom(scale).ticks(2));

        this.slider_bar_circle = this.svg.append('circle').datum(d)
            .attr('r', circle_r)
            .attr('cx', (d) => {
                return scale(d.value); //d.x - width/2 + d.width/2;
            })
            .attr('cy', (d) => {
                return d.y - d.height + 4;
            })
            .attr('fill', 'white')
            .attr('stroke', 'gray');


        this.slider_bar_circle
            .call(d3.drag()
                .on('drag', (_, ix, nds) => {
                    d3.select(nds[ix])
                        .attr('cx', (d) => {
                            return Math.max(d.x - width / 2 + d.width /
                                2, Math.min(d.x + width / 2 + d
                                    .width / 2, d3.mouse(nds[ix])[
                                        0]));
                        });

                    let v = scale.invert(Math.max(d.x - width / 2 + d.width /
                        2, Math.min(d.x + width / 2 + d.width /
                            2, d3.mouse(nds[ix])[0])));

                    this.rectDragEvent(i, v, nodes);
                })
                .on('end', (_, ix, nds) => {
                    let v = scale.invert(Math.max(d.x - width / 2 + d.width /
                        2, Math.min(d.x + width / 2 + d.width /
                            2, d3.mouse(nds[ix])[0])));

                    this.rectDragEvent(i, v, nodes);

                    this.slider_bar_circle.remove();
                    this.slider_bar_background.remove();
                    this.slider_bar.remove();
                    this.slider_bar_axis.remove();
                    this.freeze_flag.flag = false;

                    this.attUpdate();
                }));
    }

    rectDragEvent(i, d, nodes) {

        let row = Math.floor(i / this.normAttention[0].length);
        let col = i % this.normAttention[0].length;

        //renormalize current row.

        this.normAttentionRow[row][col] = d;
        this.normAttentionCol[row][col] = d;
        //this.aggregatedMatrix[row] =
        //TODO: this may be a bug if you try to renormalize the the matrix after collaspe
        this.normAttentionRow[row] = this.normalization(this.normAttentionRow[
            row]);
        // this.normAttentionCol =
        this.normalizeCol(this.normAttentionCol,
            col);
        // if (this.attentionDirection === 'row')
        //     this.normAttention = this.normAttentionRow;
        // else
        //     this.normAttention = this.normAttentionCol;

        d3.selectAll(nodes).style('fill', (d, i) => {
            let r = Math.floor(i / this.normAttention[0].length);
            let c = i % this.normAttention[0].length;

            d.value = this.normAttention[r][c];
            return this.colorbar.lookup(d.value);
        });
    }

    collapse(i, nodes, orientation) {
        d3.select(nodes[i])
            .classed('attentionMatrixComponent_text_normal', !
                d3.select(nodes[i]).classed(
                    "attentionMatrixComponent_text_normal")
            )
            .classed(
                'attentionMatrixComponent_text_collapse', !
                d3.select(nodes[i]).classed(
                    "attentionMatrixComponent_text_collapse"
                ));

        if (orientation == 'vertical') {
            this.aggregationMatrix(i, this.targ_dep.getChild(i));
            this.targ_dep.collapse(i);

        } else if (orientation == 'horizontal') {
            this.src_dep.collapse(i);
        }

        //diable animation, directly draw (shusen)
        this.draw();
        //update dependency tree position
        this.srcWords = this.collapSenBySet(this.sen2words(this.data[
            "currentPair"]["sentences"][0]), this.srcIndexMaskSet);
        this.targWords = this.collapSenBySet(this.sen2words(this.data[
            "currentPair"]["sentences"][1]), this.targIndexMaskSet);
        this.computeWordPosition(this.srcWords, this.targWords);
        this.drawDepTree();

        // this.collapse_Animation(orientation);
    }

    highlight(opt, row, col) {

        let animationTime = 100;
        if (opt == 'clean') {

            this.targtext
                .classed('attentionMatrixComponent_text_normal', (d, i,
                    nodes) => {
                    return d3.select(nodes[i]).classed(
                        "attentionMatrixComponent_text_collapse"
                    ) ? false : true;
                })
                .classed('attentionMatrixComponent_text_highlight', false);

            this.srctext
                .classed('attentionMatrixComponent_text_normal', (d,
                    i, nodes) => {
                    return d3.select(nodes[i]).classed(
                        "attentionMatrixComponent_text_collapse"
                    ) ? false : true;
                })
                .classed('attentionMatrixComponent_text_highlight', false);

            this.rectCell.transition().duration(animationTime)
                .style('opacity', 1.0);

            if (this.targ_dep)
                this.targ_dep.highlight(row);
            if (this.src_dep)
                this.src_dep.highlight(col);
        } else {

            let targWords = this.sen2words(this.data["currentPair"][
                "sentences"
            ][1]);

            this.rectCell.transition().duration(animationTime).
            style('opacity', (data, index) => {
                if (col == index % targWords.length || row ==
                    Math.floor(index / targWords.length)) {
                    return 1.0;
                } else {
                    return 0.7;
                }
            });

            this.targtext
                .classed('attentionMatrixComponent_text_normal', (d, i,
                    nodes) => {
                    if (d3.select(nodes[i]).classed(
                            "attentionMatrixComponent_text_collapse"
                        )) return false;
                    return i == col ? false : true;
                })
                .classed('attentionMatrixComponent_text_highlight', (
                    d, i, nodes) => {
                    if (d3.select(nodes[i]).classed(
                            "attentionMatrixComponent_text_collapse"
                        )) return false;
                    return i == col ? true : false;
                });

            this.srctext
                .classed('attentionMatrixComponent_text_normal', (d,
                    i, nodes) => {
                    if (d3.select(nodes[i]).classed(
                            "attentionMatrixComponent_text_collapse"
                        )) return false;
                    return i == row ? false : true;
                })
                .classed('attentionMatrixComponent_text_highlight', (
                    d, i, nodes) => {
                    if (d3.select(nodes[i]).classed(
                            "attentionMatrixComponent_text_collapse"
                        )) return false;
                    return i == row ? true : false;
                });

            if (this.targ_dep)
                this.targ_dep.highlight(col);
            if (this.src_dep)
                this.src_dep.highlight(row);
        }
    }

    handleHighlightEvent(srcIndex, targIndex) {
        if (srcIndex === -1 || targIndex === -1) {
            this.highlight('clean', srcIndex, targIndex);
        } else {
            this.highlight('highlight', srcIndex, targIndex);
        }
    }


    //define each rect's width, height, x, y and color map value
    generateMatrixGeometry() {

        let attMatrix = this.aggregatedMatrix;

        // let targWords = this.sen2words(this.data["currentPair"]["sentences"]
        //     [1]);
        //
        // let srcWords = this.sen2words(this.data["currentPair"]["sentences"]
        //     [0]);

        let w = this.width * 3 / 4 / (this.targWords.length - this.targIndexMaskSet
            .size);

        let h = this.height * 3 / 4 / (this.srcWords.length - this.srcIndexMaskSet
            .size);

        let attList = [];

        //init x location of visualization
        let attx = this.width * 1 / 4;

        //init y location of visualization
        let atty = this.height * 1 / 4;

        //row
        for (let i = 0; i < attMatrix.length; i++) {
            attx = this.width * 1 / 4;
            //column
            //init location, width and height value of each rect in the heatmap matrix
            for (let j = 0; j < attMatrix[i].length; j++) {
                let item = {
                    'x': attx,
                    'y': atty,
                    'value': attMatrix[i][j]
                };
                if (!this.srcIndexMaskSet.has(i) && !this.targIndexMaskSet.has(
                        j)) {
                    item['width'] = w;
                    item['height'] = h;
                    attx += w;
                } else {
                    item['width'] = 0;
                    item['height'] = 0;
                }
                attList.push(item);
            }

            //if current row is not filtered
            if (!this.srcIndexMaskSet.has(i)) {
                atty += h;
            }
        }

        return attList;
    }

    //define each text font x, y, and font-size base on whether they are filtered
    generateTextGeometry() {

        let srcText = [];

        // let srcWords = this.sen2words(this.data["currentPair"]["sentences"][0]);

        let h = (this.height * 0.75) / (this.srcWords.length - this.srcIndexMaskSet
            .size);

        let text_loc = h / 2 + this.height / 4;
        for (let i = 0; i < this.srcWords.length; i++) {
            let item = {};

            item['x'] = this.width * 1 / 4 - this.margin.left * 3;

            item['y'] = text_loc;

            item['text'] = this.srcWords[i];

            if (!this.srcIndexMaskSet.has(i)) {
                item['display'] = 'block';
                text_loc += h;
            } else {
                item['display'] = 'none';
            }
            srcText.push(item);
        }

        let targText = [];

        // let targWords = this.sen2words(this.data["currentPair"]["sentences"]
        //     [1]);

        let w = (this.width * 0.75) / (this.targWords.length - this.targIndexMaskSet
            .size);

        text_loc = w / 2 + this.width / 4;
        for (let i = 0; i < this.targWords.length; i++) {
            let item = {};

            item['x'] = text_loc;

            item['y'] = this.height * 1 / 4 - this.margin.top * 3;

            item['text'] = this.targWords[i];

            if (!this.targIndexMaskSet.has(i)) {
                item['display'] = 'block';
                text_loc += w;
            } else {
                item['display'] = 'none';

            }
            targText.push(item);
        }

        return {
            'src': srcText,
            'targ': targText
        };

    }

    collapse_Animation(direction) {

        //////////////////// rect animation ////////////////////
        let attMatrix = this.generateMatrixGeometry();

        let rects = d3.selectAll('.attentionComponent_matrix_rect').data(
            attMatrix);

        rects.exit().remove();

        rects.append('rect');

        //if(direction == 'horizontal'){
        rects
            .transition()
            .duration(1000)
            .attr('x', (d, i) => {
                return d.x;
            })
            .attr('y', (d, i) => {
                return direction == 'vertical' && d.height == 0 ? this.height /
                    4 : d.y;
            })
            .attr('width', (d) => {
                return d.width;
            })
            .attr('height', (d) => {
                return d.height;
            })
            .style('fill', d => {
                return this.colorbar.lookup(d.value);
            });

        let texts = this.generateTextGeometry();
        /////////////////  src text animation ///////////////////
        let srctext = this.svg.selectAll('.attentionComponent_srcWords')
            .data(texts.src);

        srctext.exit().remove();
        srctext.append('text')

        srctext
            .transition()
            .duration(1000)
            .attr('x', (d, i) => d.x)
            .attr("y", (d, i) => d.y)
            .attr("display", (d) => d['display']);

        /////////////////  targ text animation ///////////////////
        let targtext = this.svg.selectAll('.attentionComponent_targWords')
            .data(texts.targ);

        targtext.exit().remove();
        targtext.append('text');

        targtext
            .transition()
            .duration(1000)
            .attr('x', (d, i) => d.x)
            .attr("y", (d, i) => d.y)
            .attr("transform", (d, i) => {
                return "rotate(-45, " + d.x + ' , ' + d.y + ')';
            })
            .attr("display", (d) => d['display']);


        ///////////////// dependency tree animation ///////////////////
        this.srcWords = this.collapSenBySet(this.sen2words(this.data[
            "currentPair"]["sentences"][0]), this.srcIndexMaskSet);
        this.targWords = this.collapSenBySet(this.sen2words(this.data[
            "currentPair"]["sentences"][1]), this.targIndexMaskSet);
        this.computeWordPosition(this.srcWords, this.targWords);
        this.drawDepTree();
    }

    updateMatrixColormap() {
        if (this.svg) {
            this.svg.selectAll('.attentionComponent_matrix_rect')
                .data(this.attList)
                .style('fill', d => {
                    // console.log(d);
                    return this.colorbar.lookup(d.value);
                });
        }
    }

    drawDepTree() {
        if (this.srcDepTreeData) {
            if (this.src_dep === undefined) {
                this.svg.selectAll(
                    '.attentionMatrixComponent_background_text').remove();

                // if (this.src_dep)
                //     this.src_dep.clear();
                this.src_dep = new dependencyTreePlot(this.svg, 'v-left',
                    this.srcWords, this.srcPos, this.srcDepTreeData,
                    this.width, this.height);
                this.src_dep.setCollapseHandle(this.collapseSrc.bind(
                    this));
                // console.log("create tree");
            } else {
                this.src_dep.updatePos(this.srcPos);
            }
        }

        if (this.targDepTreeData) {
            if (this.targ_dep === undefined) {

                // if (this.targ_dep)
                //     this.targ_dep.clear();
                this.targ_dep = new dependencyTreePlot(this.svg, 'h-top',
                    this.targWords, this.targPos, this.targDepTreeData,
                    this.width, this.height);
                this.targ_dep.setCollapseHandle(this.collapseTarget.bind(
                    this));

            } else {
                // this.targ_dep.setData(this.targDepTreeData);
                this.targ_dep.updatePos(this.targPos);
            }
        }
    }

    computeWordPosition(src, targ) {
        // console.log(src, targ);
        this.srcPos = this.srcWords.map((d, i) => {
            return {
                'y': (this.height * 0.75) / this.srcWords.length *
                    (i + 0.5) + this.height / 4,
                'x': this.width * 1 / 4 - this.margin.left * 3
            };
        });

        this.targPos = this.targWords.map((d, i) => {
            return {
                'x': (this.width * 0.75) / this.targWords.length *
                    (i + 0.5) + this.width / 4,
                'y': this.height * 1 / 4 - this.margin.top * 3
            };
        });
    }

}
