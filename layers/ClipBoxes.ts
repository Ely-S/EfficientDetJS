import * as tf from '@tensorflow/tfjs';
import {layers, Shape, Tensor, tidy} from '@tensorflow/tfjs';

export class ClipBoxes extends layers.Layer {
  /** @nocollapse */
  static className = 'ClipBoxes';

  computeOutputShape(inputShape: Shape[]): Shape|Shape[]{return inputShape[1]}

  call(inputs: Tensor): Tensor|Tensor[] {
        return tidy(() => {
            let image = inputs.gather(0)
        let boxes = inputs.gather(1)

        let maxHeight = image.shape[1] - 1
        let maxWidth = image.shape[2] - 1

        let x1 = tf.clipByValue(boxes.$(':, :, 0'), 0, maxWidth)
        let y1 = tf.clipByValue(boxes.$(':, :, 1'), 0, maxHeight)
        let x2 = tf.clipByValue(boxes.$(':, :, 2'), 0, maxWidth)
        let y2 = tf.clipByValue(boxes.$(':, :, 3'), 0, maxHeight)

            return tf.stack([x1, y1, x2, y2], 2)
        });
  }
}
