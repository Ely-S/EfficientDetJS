import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import { Shape, LayerVariable, Tensor } from "@tensorflow/tfjs";

export declare interface wBiFPNAddArgs extends LayerArgs {
    epsilon?: number
}

export class wBiFPNAdd extends tf.layers.Layer {
    static className = 'wBiFPNAdd';

    private epsilon: number
    private w: LayerVariable
    // layer weights is named w for consistency with the python implementaiton

    constructor(args: wBiFPNAddArgs) {
        super(args)
        this.epsilon = args.epsilon || 1E-4
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
    }

    public build(inputShape: Shape): void {
        const num_in = inputShape.length
        const initializer = tf.initializers.constant({ value: 1 / num_in })
        const dtype = 'float32'
        const shape = [num_in]
        const regularizer = null
        const trainable = true

        this.w = this.addWeight(
            this.name,
            shape,
            dtype,
            initializer,
            regularizer,
            trainable,
        )
    }


    call(inputs: Tensor[], kwargs) {
        return tf.tidy(() => {
            let w = this.w.read().relu()

            // elementwise_multiply = [w[i] * inputs[i] for i in range(len(inputs))]
            let elementwise_multiply = inputs.map((inp, index) => w.gather([index]).mul(inp))

            // stack these because inputs is an array
            let mulStack = tf.stack(elementwise_multiply, 0)

            // use .sum(0) instead of tf.reduce_sum(0)
            // x = tf.reduce_sum(elementwise_multiply, axis=0)
            let x = mulStack.sum(0)

            // x = x / (tf.reduce_sum(w) + self.epsilon)
            // addN operates on a list of tensors, not a tensor
            let denominator = w.sum(0).add(this.epsilon)

            return x.div(denominator)
        });
    }

    // Every layer needs a unique name.
    getClassName() {
        return 'wBiFPNAdd';
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        config.epsilon = this.epsilon
        return config;
    }

}
