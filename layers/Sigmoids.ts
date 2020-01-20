import * as tf from '@tensorflow/tfjs';
import { tidy } from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/src/activations';
import { getExactlyOneTensor } from '@tensorflow/tfjs-layers/src/utils/types_utils';


// 'swish' is not a supported tfjs activation function right now
export class Swish extends Activation {
  /** @nocollapse */
  static readonly className = 'swish';
  /**
   * Calculate the activation function.
   *
   * @param x: Input.
   * @return Output of the Swish activation.
   */
  apply(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tidy(() => {
      let x = getExactlyOneTensor(inputs)
      return tf.sigmoid(x).mul(x);
    })
  }
}
