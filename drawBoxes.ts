export function drawBoxes(predictions, canvas) {
    const ctx = canvas.getContext("2d");
    // ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "24px helvetica";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach(prediction => {
        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        const width = prediction.bbox[2] - x;
        const height = prediction.bbox[3] - y;

        // Draw the bounding box.
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        ctx.fillStyle = "#FF0000";
        const textWidth = ctx.measureText(prediction.class).width;
        const textHeight = parseInt(font, 10);

        // draw top left rectangle
        ctx.fillRect(x, y, textWidth + 10, textHeight + 10);

        // draw bottom left rectangle
        ctx.fillRect(x, y + height - textHeight, textWidth + 15, textHeight + 10);

        // Draw the text last to ensure  it's on top.
        ctx.fillStyle = "#ffff";
        ctx.fillText(prediction.class, x, y);
        ctx.fillText(prediction.score.toFixed(2), x, y + height - textHeight);
    });
}
