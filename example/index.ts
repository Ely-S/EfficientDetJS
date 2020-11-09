import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';


import EfficientDet from "../src/index"

window.tf = tf

tf.enableProdMode()


const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('livecanvas');
const timeEl = document.getElementById("time")

videoElement.width = videoElement.height = 1200;

var model = new EfficientDet()

var camera

async function loop() {
  let start = Date.now()

  let img = await camera.capture()

  console.time("Prediction")
  let boxes = await model.predict(img)
  console.timeEnd("Prediction")

  tf.dispose(img)

  window.requestAnimationFrame(() => {
    // draw predictions on the canvas
    model.drawBoxes(boxes, liveCanvasElement, true)

    // report approximate time to display predictions
    timeEl.innerText = Date.now() - start + "ms "

    window.requestAnimationFrame(loop)
  })
}


async function start() {
  await tf.ready()
  console.log("Initialized ", tf.backend())

  // start loading the model
  const modelLoaded = model.load()

  console.log("getting camera")
  camera = await tf.data.webcam(videoElement, {
    facingMode: 'environment',
    centerCrop: false
  });

  let img = await camera.capture()

  let [height, width] = img.shape
  videoElement.height = liveCanvasElement.height = height
  videoElement.width = liveCanvasElement.width = width
  timeEl.style.marginTop = (height + 50) + "px"

  console.log("loop")
  await modelLoaded
  document.getElementById("loading").hidden = true

  loop()
}



start()