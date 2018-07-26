/*
 - standardize query interface
 - handle component wise communication
*/

// console.log(document.domain, location.port);

class baseComponent {
    constructor(uuid) {
        this.uuid = uuid;
        // console.log(this.uuid);
        this.div = "#" + this.uuid;
        this.data = {};

        socket.on(this.uuid, this.parseMessage.bind(this));

        //default margin
        this.margin = {
            top: 5,
            right: 5,
            bottom: 5,
            left: 5
        };

        this.calledFunc = Object();
        // Define the div for the tooltip
        // console.log(d3.select(this._getContainer().get()));
        if (!d3.select(this.div + "container").empty()) {
            this.tooltip = d3.select(d3.select(this.div + "container").node()
                    .parentNode)
                // this.tooltip = d3.select(this.div + "container")
                .append("div")
                .attr("class", "notice")
                .style("position", "relative")
                .style("display", "none")
                .style("opacity", 0.5);
        }
    }

    showTooltip(pos, message) {

        this.tooltip.html(message)
            .style("display", "inline-block")
            .style("position", "relative")
            .style("left", pos[0] + "px")
            .style("top", -35 + pos[1] + "px");

    }

    subscribeDatabyNames(names) {
        if (!Array.isArray(names)) {
            console.log("Error: input need to be a list of names\n");
            return;
        }

        for (var i = 0; i < names.length; i++) {

            var msg = {
                "type": "subscribeData",
                "name": names[i],
                "uid": this.uuid
            };
            // console.log(msg);
            socket.emit('message', msg);
        }
    }

    callFunc(funcName, params = {}) {
        var msg = {
            "type": "call",
            "func": funcName,
            "params": params,
            "uid": this.uuid
        };
        socket.emit('message', msg);
        this.addCallSet(funcName);
        this.updateServerStateDisplay();
    }

    updateServerStateDisplay() {
        // console.log(this.calledFunc);
        // Object.keys(this.calledFunc)
        var allReturned = true;
        for (var key in this.calledFunc) {
            if (this.calledFunc.hasOwnProperty(key)) {
                if (this.calledFunc[key] !== 0) {
                    this.showTooltip(
                        [5.0, 0.0],
                        // [this.width * 0.5, 0.0],
                        // [this.width * 0.5, this.height * 0.5 ],
                        "Running:" + key + "...");
                    allReturned = false;
                    // console.log(key, this.calledFunc[key]);
                }

            }
        }
        if (allReturned)
            this.tooltip.html("").style("display", "none");

    }

    addCallSet(funcName) {
        if (this.calledFunc[funcName]) {
            this.calledFunc[funcName] += 1;
        } else {
            this.calledFunc[funcName] = 1;
        }
    }

    removeCallSet(funcName) {
        if (this.calledFunc[funcName]) {
            if (this.callFunc[funcName] === 0)
                delete this.calledFunc[funcName];
            else
                this.calledFunc[funcName] -= 1;
        }
        this.updateServerStateDisplay();
    }

    setData(name, data) {
        this.data[name] = data;
        var msg = {
            "type": "setData",
            "name": name,
            "data": data,
            "uid": this.uuid
        };
        // console.log(msg);
        socket.emit('message', msg);
    }

    parseMessage(msg) {
        // console.log("\nparse message in base class\n", msg);
        switch (msg['type']) {
            case 'data':
                this.parseDataUpdate(msg);
                break;
            case 'funcReturn':
                this.parseFunctionReturn(msg);
                break;
        }
    }

    parseFunctionReturn(msg) {
        this.removeCallSet(msg['func']);
    }

    parseDataUpdate(msg) {
        this.updateData(msg);
    }

    updateData(msg) {
        var name = msg["name"];
        var data = msg["data"]["data"];
        this.data[name] = data;
    }

    ////////// implemented by individual component ////////

    draw() {

    }

    resize() {

    }


    /////////// helper function //////////////
    _getContainer() {
        return $(this.div).parent().parent().parent();
    }
    _updateWidthHeight() {
        //resize width height
        //parent width, height
        this.pwidth = $(this.div).parent().parent().parent().width();
        this.pheight = $(this.div).parent().parent().parent().height();
        // console.log(this.pwidth, this.pheight);

        //setup single plot data
        this.width = this.pwidth - this.margin.left - this.margin.right;
        this.height = this.pheight - this.margin.top - this.margin.bottom;
        // console.log(this.width, this.height);
    }

}
