import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';


import EfficientDet from "../src/index"

window.tf = tf

tf.enableProdMode()


const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('livecanvas');
const timeEl = document.getElementById("time")

videoElement.width = videoElement.height = 1200;

var model = new EfficientDet({})

async function loop() {
  let img = await capturePhoto()


  let start = Date.now()
  console.time("Prediction")
  let boxes = await model.predict(img)
  console.timeEnd("Prediction")

  tf.dispose(img)

  window.requestAnimationFrame(async () => {

    model.drawBoxes(boxes, liveCanvasElement, true)
    timeEl.innerText = Date.now() - start + "ms "

    window.requestAnimationFrame(loop)
  })
}


var camera

async function capturePhoto(): Promise<Tensor3D> {
  return (await camera).capture()
}
async function start() {
  await tf.ready()
  console.log("Initialized ", tf.backend())

  console.log("getting camera")
  camera = tf.data.webcam(videoElement, {
    facingMode: 'environment',
    centerCrop: false
  });

  let img = await capturePhoto()

  let [height, width] = img.shape
  videoElement.height = liveCanvasElement.height = height
  videoElement.width = liveCanvasElement.width = width
  timeEl.style.marginTop = (height + 50) + "px"

  await model.load()

  document.getElementById("loaidng").hidden = true

  console.log("loop")
  loop()
}



start()