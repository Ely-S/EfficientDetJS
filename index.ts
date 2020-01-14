import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';

import { PriorProbability } from './initializers';
import { ClipBoxes } from './layers/ClipBoxes';
import { FilterDetections } from './layers/FilterDetections';
import { RegressBoxes } from './layers/RegressBoxes';
import { Reshape } from './layers/Reshape';
import { SigmoidLayer, Swish, SwishLayer } from './layers/Sigmoids';

window.tf = tf

tf.enableProdMode()
tf.setBackend('webgl')

tf.serialization.registerClass(Swish);
tf.serialization.registerClass(RegressBoxes);
tf.serialization.registerClass(Reshape);
tf.serialization.registerClass(PriorProbability)
tf.serialization.registerClass(ClipBoxes)
tf.serialization.registerClass(FilterDetections)
tf.serialization.registerClass(SwishLayer)
tf.serialization.registerClass(SigmoidLayer)

const camConfig = {
  // facingMode: 'user',
  resizeWidth: 640,
  resizeHeight: 640,
  centerCrop: true
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const canvasElement = <HTMLCanvasElement>document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = 640;
canvasElement.height = 640;

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D
  tf.browser.toPixels(scaledImage, canvasElement)

  return scaledImage
}

function drawBoxes(predictions, canvas) {
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

    console.log(x, y, height, width)

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
};

async function start() {
  await tf.ready()
  let model = await tf.loadLayersModel('/pascal_phi1_unweighted/model.json')
  // '/pascal_unweighted_sigmoidlayer_swishlayer_nofilter/model.json')

  window.model = model

  model.summary()


  let scaledImage = await capturePhoto() as Tensor3D
  let batch = scaledImage.expandDims()

  console.log('DONE LOADING')
  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423
  model.predict([batch], { verbose: true })

  console.log('done predicting')

  for (let i = 0; i < 2; i++) {
    console.time("Prediction")
    let p = model.predict([batch])
    console.timeEnd("Prediction")

    let allBoxes = p[0]
    let allScores = p[1]

    // allBoxes has shape (batch_size, boxes, 4)
    // allScores has shape (batch_size, boxes, n_classes)
    // where boxes = n_ancor_boxes

    // perform inference on one image at a time

    let boxes = allBoxes.gather(0).squeeze()
    let scores = allScores.gather(0).squeeze()


    let maxOutPutSize = 2
    let iouThreshold = .5
    let scoreThreshold = .5

    console.log(boxes)


    // person is class 14
    console.time("nms")

    let scores_for_class_i = scores.$(":," + 14)

    let indecies = await tf.image.nonMaxSuppressionAsync(
      boxes, scores_for_class_i,
      maxOutPutSize, iouThreshold, scoreThreshold
    )

    indecies.print()
    let pickedBoxes = boxes.gather(indecies)
    pickedBoxes.print()

    let predBoxes = pickedBoxes.dataSync()
    let predScores = scores_for_class_i.gather(indecies).dataSync()

    console.log(predBoxes)
    console.log(predScores)

    console.timeEnd("nms")

    for (let i = 0; i < predScores.length; i++) {
      drawBoxes([{
        bbox: predBoxes.slice(i * 4, i * 4 + 4),
        class: "human",
        score: predScores[i]
      }], canvasElement)
    }

    console.log("draw end")


  }

  // function nms_by_class(){
  //   for (let i = 0; i < n_classes; i++) {
  //     // let scores_for_class_i = scores.slice([0, i], [-1, i + 1]).squeeze()
  //     let scores_for_class_i = scores.$(":," + i)

  //     //@TODO: use nonMaxSuppressionWithScoreAsync
  //     indecesPromises[i] = tf.image.nonMaxSuppressionAsync(
  //       boxes, scores_for_class_i,
  //       maxOutPutSize, iouThreshold, scoreThreshold
  //     )

  //   }

  // }

}


start()