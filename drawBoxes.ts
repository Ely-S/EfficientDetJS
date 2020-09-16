export function drawBoxes(predictions, canvas) {
    const ctx = canvas.getContext("2d");
    // ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "8px helvetica";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach(prediction => {
        let [x, y, width, height] = prediction.bbox


        // Draw the bounding box.
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        ctx.fillStyle = "#FF0000";
        const textWidth = ctx.measureText(prediction.class).width;
        const textHeight = parseInt(font, 10);

        // draw top left rectangle
        ctx.fillRect(x, y, textWidth, textHeight);

        // draw bottom left rectangle
        ctx.fillRect(x, y + height - textHeight, textWidth, textHeight);

        // Draw the text last to ensure  it's on top.
        ctx.fillStyle = "#ffff";
        ctx.fillText(prediction.class, x, y);
        ctx.fillText(prediction.score.toFixed(2), x, y + height - textHeight);
    });
}
