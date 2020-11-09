
import * as tf from '@tensorflow/tfjs';
import { Tensor3D, GraphModel, Tensor4D } from '@tensorflow/tfjs';

// Labels for the COCO dataset
const labels = {
  1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
  6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
  11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
  16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
  22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
  28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
  35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
  39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
  43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
  49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
  54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
  59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
  64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
  73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
  78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
  84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
  89: 'hair drier', 90: 'toothbrush',
}

interface EfficientDetOptions {
}

interface Box {
  score: number,
  class: string,
  bbox: {
    x: number,
    y: number,
    width: number,
    height: number
  }
}

export default class EfficientDet {
  labels: { [index: number]: string } = labels
  modelURI = 'd0/model.json'
  model?: GraphModel
  minScore = 0.01

  constructor(options?: EfficientDetOptions) {

  }

  load = async () => {
    await tf.ready()
    this.model = await tf.loadGraphModel(this.modelURI)
  }

  predict = async (image: Tensor3D) => {
    let batch: Tensor4D = image.expandDims()

    try {
      return await this.predictBatch(batch)
    } finally {
      tf.dispose(batch)
    }
  }


  predictBatch = async (imageBatch: Tensor4D): Promise<Box[]> => {
    if (!this.model) {
      throw new Error("Model not loaded yet, call .load()")
    }

    // input imageBatch to placeholder image_arrays
    // get output tensor 'detections'
    let detections = await this.model.executeAsync(
      { image_arrays: imageBatch }, "detections") as tf.Tensor

    let predictions = await detections.array()[0] as Array<Array<number>>

    tf.dispose(detections)

    let detectedObjects: Box[] = []

    // There may be no predictions, which would result in undefined
    if (predictions === undefined) return detectedObjects

    predictions.forEach(out => {
      let [image_id, y, x, ymax, xmax, score, _class] = out

      if (score < this.minScore) return
      detectedObjects.push({
        bbox: {
          x: x,
          y: y,
          width: xmax - x,
          height: ymax - y
        },
        "class": this.labels[_class],
        score
      } as Box)
    });


    return detectedObjects
  }

  drawBoxes = (boxes: Box[], canvas: HTMLCanvasElement, clearCanvas = false) => {
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D

    if (clearCanvas) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "12px ariel";
    ctx.font = font;
    ctx.textBaseline = "top";

    boxes.forEach(prediction => {
      // let [x, y, width, height] = prediction.bbox
      let { x, y, width, height } = prediction.bbox


      // Draw the bounding box.
      ctx.strokeStyle = "#FF0000";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      var label = (prediction.score * 100).toFixed(0)
      ctx.fillStyle = "#FF0000";
      const textWidth = ctx.measureText(label).width * 1.3;
      const textHeight = parseInt(font, 10); + 2

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
}
