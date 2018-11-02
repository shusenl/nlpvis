class latentSpaceComponent extends baseComponent {
    constructor(uuid) {
        super(uuid);
        this.subscribeDatabyNames(["currentPair", "allPairsPrediction"]);

        this.margin = {
            top: 10,
            right: 10,
            bottom: 10,
            left: 10
        };

        $(this.div + "container").parent().css("overflow-y", "scroll");

        this.tableEntry = null;

        d3.select(this.div + "refresh").on("click", this.triggerLookup.bind(
            this));
    }

    parseDataUpdate(msg) {
        super.parseDataUpdate(msg);

        switch (msg['name']) {
            case "currentPair":
                break;
                // this.callFunc("latentStateLookup", {
                //     "sentence": this.data["currentPair"][
                //         "sentences"
                //     ][0]
                // });
                //FIXME only for hypothesis sentence
            case "allPairsPrediction":
                this.clear();
                //trigger lookup
                this.triggerLookup();
                break;
        }
    }

    triggerLookup() {
        console.log("trigger lookup\n");
        this.callFunc("latentStateLookup", {
            "sentence": this.data["currentPair"][
                "sentences"
            ][1]
        });
    }

    parseFunctionReturn(msg) {
        super.parseFunctionReturn(msg);

        switch (msg['func']) {
            case 'latentStateLookup':
                this.handleNeighborLookup(msg['data']["data"]);
        }
    }

    handleNeighborLookup(neighbors) {
        // console.log(neighbors);
        //convert dict to table
        if (Object.keys(neighbors).length === 0)
            return;

        let sens = neighbors["sentence"];
        let pred = neighbors["prediction"]
        let maxDist = Math.max(...neighbors["distance"]);
        let dists = neighbors["distance"].map(d => d / maxDist);
        this.tableEntry = [];
        for (var i = 0; i < sens.length; i++) {
            this.tableEntry.push([d3.format(".2f")(dists[i]), sens[i], pred[
                i]]);
        }
        // console.log(this.tableEntry);
        this.draw();
    }

    clear() {
        d3.select(this.div + "table").selectAll("*").remove();
    }

    draw() {
        //////// drawing table ////////
        d3.select(this.div + "table").selectAll("*").remove();
        var thead = d3.select(this.div + "table").append('thead');
        var tbody = d3.select(this.div + "table").append('tbody');
        // console.log(thead, tbody);
        // append the header row
        let columns = [{
            "label": "dist",
            "width": "15%"
        }, {
            "label": "sentence",
            "width": "85%"
        }];
        thead.append('tr')
            .selectAll('th')
            .data(columns).enter()
            .append('th')
            .style("width", t => t.width)
            .text(t => t.label);

        if (this.tableEntry) {
            let data = this.tableEntry;
            // create a row for each object in the data
            var rows = tbody.selectAll('tr')
                .data(data)
                .enter()
                .append('tr');

            // create a cell in each row for each column
            // let colormap = ["lightgreen", "Gainsboro", "GhostWhite"];
            var cells = rows.selectAll('td')
                .data(function(row) {
                    return columns.map(function(column, i) {
                        let entry = row[i];

                        return {
                            column: column,
                            value: entry,
                            pred: row[2]
                        };
                    });
                })
                .enter()
                .append('td')
                .style("background-color", d => {
                    if (d.pred)
                        return "lightgreen";
                })
                .text(d => d.value);
        }
    }

}
