export function drawBoxes(predictions, canvas, clear = true) {
    const ctx = canvas.getContext("2d");
    if (clear) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "15px ariel";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach(prediction => {
        let [x, y, width, height] = prediction.bbox


        // Draw the bounding box.
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        var label = (prediction.score * 100).toFixed(1) + "%"
        ctx.fillStyle = "#FF0000";
        const textWidth = ctx.measureText(label).width + 2;
        const textHeight = parseInt(font, 10); + 4

        // draw top left rectangle
        ctx.fillRect(x, y - 2, textWidth, textHeight);

        // draw bottom left rectangle
        ctx.fillRect(x, y + height - textHeight, textWidth, textHeight);

        // Draw the text last to ensure  it's on top.
        ctx.fillStyle = "#ffff";
        ctx.fillText(prediction.class, x, y);
        ctx.fillText(label, x, y + height - textHeight);
    });
}
