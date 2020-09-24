import * as tf from '@tensorflow/tfjs';
import { Tensor3D, Tensor2D, Tensor1D, Tensor4D, Tensor } from '@tensorflow/tfjs';


function get_1d_gaussian_kernel(sigma, size): Tensor1D {
    var x = tf.range(Math.floor(-size / 2) + 1, Math.floor(size / 2) + 1)
    x = tf.pow(x, 2)
    x = tf.exp(x.div(-2.0 * (sigma * sigma))) as Tensor1D
    x = x.div(tf.sum(x))
    return x
}

function get_2d_gaussian_kernel(size: number, sigma?: number): Tensor2D {
    // // This is to mimic opencv2. 
    sigma = sigma || (0.3 * ((size - 1) * 0.5 - 1) + 0.8)

    var d1 = get_1d_gaussian_kernel(sigma, size)
    return tf.outerProduct(d1, d1)
}

export function getGaussianKernel(size = 5): Tensor4D {
    return tf.tidy(() => {
        var d2 = get_2d_gaussian_kernel(size)
        var d3 = tf.stack([d2, d2, d2])
        return tf.reshape(d3, [size, size, 3, 1])
    })
}

export function blur(image: Tensor3D, kernel: Tensor4D): Tensor3D {
    return tf.tidy(() => {
        return tf.depthwiseConv2d(image, kernel, 1, "valid")
    })
}

