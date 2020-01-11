import * as tf from '@tensorflow/tfjs';

import { Initializer } from '@tensorflow/tfjs-layers/src/initializers';
import { Shape, DataType, Tensor } from '@tensorflow/tfjs';

interface PriorProbabilityArgs {
    probability: number
}

export class PriorProbability extends Initializer {
    static readonly className = 'PriorProbability';
    private probability: number

     constructor(args: PriorProbabilityArgs) {
         super()
         this.probability = args.probability
     }

    apply(shape: Shape, dtype?: DataType): Tensor {
        let scalar = -Math.log((1 - this.probability) / this.probability)
        return tf.ones(shape, dtype).mul(scalar)
    }

    getConfig(): tf.serialization.ConfigDict {
        return {probability: this.probability};
    }  
}
