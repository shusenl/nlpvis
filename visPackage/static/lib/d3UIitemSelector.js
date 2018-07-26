class d3UIitemSelector {
    constructor(svg, list, initIndex = 0, pos = [15, 15], size = [60, 20],
        maxWidth = "100px") {
        // this.svgTag = "#" + svgTag;
        this.svgContainer = svg;
        this._pos = pos;
        this._size = size;
        this._list = list;
        // console.log(list);
        // this._list = JSON.parse(JSON.stringify(list)); //list to select from
        this.id = 'd3UIitemSelector-' + uuidv1();
        this._index = initIndex;
        this.maxWidth = maxWidth;

        this.draw();
    }

    setTempLabel(label) {
        this._tempLabel = label;
        this.draw();
    }

    draw() {
        var htmlList = "";
        for (var i = 0; i < this._list.length; i++) {
            if (i === Number(this._index))
                htmlList += "<option value=" + i + " selected>" + this._list[
                    i] +
                "</option>";
            else
                htmlList += "<option value=" + i + ">" + this._list[i] +
                "</option>";
        }

        if (this._tempLabel) {
            htmlList += "<option selected>" + this._tempLabel +
                "</option>";
        }

        var htmlSelector = "<select id=" + this.id + ">" + htmlList +
            "<select/>";

        //cleanup
        this.svgContainer.select("#" + this.id + "_fobject").remove();
        //add new
        this.svgContainer.append("foreignObject")
            .attr('id', this.id + "_fobject")
            .attr("x", this._pos[0])
            .attr("y", this._pos[1])
            .attr("width", this._size[0])
            .attr("height", this._size[1])
            // .append("xhtml:body") //this will cause issues in different browsers
            .style("font", "12px")
            .html(htmlSelector);


        d3.select('#' + this.id).style("max-width", this.maxWidth);

        this.svgContainer.select('#' + this.id)
            .on('change', this.selectionChanged.bind(this));
    }

    // triggerDropdown(){
    //   $('#'+this.id).trigger('click');
    // }
    pos(pos) {
        //reset state
        this._pos = pos;
        // this._tempLabel = undefined;
        // this.draw();
        this.svgContainer.select("#" + this.id + "_fobject")
            .attr("x", this._pos[0])
            .attr("y", this._pos[1])
            // this.svgContainer.attr("x", this._pos[0]).attr("y", this._pos[1]);
    }

    selection() {
        // this._index = this.svgContainer.select('#' + this.id).node().value;
        return this._index;
    }

    selectionLable() {
        this._index = this.svgContainer.select('#' + this.id).node().value;
        if (this._index) {
            return this._list[this._index];
        }
    }

    selectionChanged() {
        this._index = this.svgContainer.select('#' + this.id).node().value;
        console.log("Index changed to: ", Number(this.selection()));
        if (this.updateCallback)
            this.updateCallback(Number(this.selection()));

        //reset the entry if there is temp label
        if (this._tempLabel) {
            this._tempLabel = undefined;
            this.draw();
        }
    }

    callback(func) {
        this.updateCallback = func;
    }

    updateSelectList(names) {
        this._list = names;
        this.draw();
    }

    setCurrentName(name) {
        for (var i = 0; i < this._list.length; i++) {
            if (name === this._list[i]) {
                this._index = i;
                // console.log(i);
            }
        }
        this.draw();
    }
}
