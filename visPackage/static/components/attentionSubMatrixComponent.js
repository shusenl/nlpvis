class attentionSubMatrixComponent extends attentionMatrixComponent {
    constructor(uuid) {
        super(uuid);
        //sub matrix
        this.subscribeDatabyNames(["selectionRange"]);
        this.backgourndText = ['Context', 'Question'];
    }

    parseDataUpdate(msg) {
        // super.parseDataUpdate(msg);
        //skip the base class's parse operation
        this.updateData(msg);
        switch (msg["name"]) {
            case "selectionRange":
                this.draw();
                break;
            case "attention":
                if (this.rawAttention) {
                    //clone the raw attention
                    this.preRawAtt = JSON.parse(JSON.stringify(this.rawAttention));
                }
                this.rawAttention = this.data["attention"];
                this.attentionDirection = 'row';
                this.normAttentionRow = this.convertRawAtt(this.rawAttention,
                    'row');

                this.normAttentionCol = this.convertRawAtt(this.rawAttention,
                    'col');

                this.normAttention = this.normAttentionRow;
                break;
        }

    }

    // resize() {
    //
    // }

}
