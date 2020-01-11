import * as tf from '@tensorflow/tfjs';
import {Shape, Tensor, tidy} from '@tensorflow/tfjs';
import {Activation} from '@tensorflow/tfjs-layers/src/activations';


// 'swish' is not a supported tfjs activation function right now
export class SwishLayer extends tf.layers.Layer {
  /** @nocollapse */
  static className = 'SwishLayer';

  computeOutputShape(inputShape: Shape[]): Shape|Shape[]{return inputShape}

  call(inputs: Tensor): Tensor|Tensor[] {
    return tidy(() => {return inputs.mul(tf.sigmoid(inputs))});
  }
}

export class Swish extends Activation {
  /** @nocollapse */
  static readonly className = 'swish';
  /**
   * Calculate the activation function.
   *
   * @param x: Input.
   * @return Output of the Swish activation.
   */
  apply(x: tf.Tensor): tf.Tensor {
    return tf.sigmoid(x).mul(x);
  }
}


// tf.layers.Activation('sigmoid') doesn't port to tfjs for some reason
// so it is implemented here.
export class SigmoidLayer extends tf.layers.Layer {
  /** @nocollapse */
  static className = 'SigmoidLayer';

  computeOutputShape(inputShape: Shape[]): Shape|Shape[]{return inputShape}

  call(inputs: Tensor): Tensor|Tensor[] {
    return tf.sigmoid(inputs)
  }
}
