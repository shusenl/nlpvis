/*

Density overaly for the prediction view

*/

class densityOverlayPlot {
    constructor() {

    }

    draw() {
        svg.insert("g", "g")
            .attr("fill", "none")
            .attr("stroke", "#000")
            .attr("stroke-width", 0.5)
            .attr("stroke-linejoin", "round")
            .selectAll("path")
            .data(d3.contourDensity()
                .x(function(d) {
                    return x(d[0]);
                })
                .y(function(d) {
                    return y(d[1]);
                })
                .size([width, height])
                .bandwidth(10)
                (this.data))
            .enter().append("path")
            .attr("fill", function(d) {
                return color(d.value);
            })
            .attr("d", d3.geoPath());
    }
}
