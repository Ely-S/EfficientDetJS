import * as tf from '@tensorflow/tfjs';
import { layers, Shape, Tensor, tidy } from '@tensorflow/tfjs';
import { EasyTensor } from './hacks';

export class ClipBoxes extends layers.Layer {
  /** @nocollapse */
  static className = 'ClipBoxes';

  computeOutputShape(inputShape: Shape[]): Shape | Shape[] { return inputShape[1] }

  call(inputs: Tensor[]): Tensor | Tensor[] {
    return tidy(() => {

      // Batch of images with shape (n, height, width, channels) 
      let images = inputs[0]

      // Batch of predicted boxes with shape (n, n_predictions, 4)
      let boxes = inputs[1] as EasyTensor

      let maxHeight = images.shape[1] - 1
      let maxWidth = images.shape[2] - 1

      let x1 = tf.clipByValue(boxes.$(':, :, 0'), 0, maxWidth)
      let y1 = tf.clipByValue(boxes.$(':, :, 1'), 0, maxHeight)
      let x2 = tf.clipByValue(boxes.$(':, :, 2'), 0, maxWidth)
      let y2 = tf.clipByValue(boxes.$(':, :, 3'), 0, maxHeight)
      return tf.stack([x1, y1, x2, y2], 2)
    });
  }
}
