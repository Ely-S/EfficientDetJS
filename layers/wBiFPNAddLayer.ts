import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import { Shape, LayerVariable } from "@tensorflow/tfjs";

export declare interface wBiFPNAddArgs extends LayerArgs {
    epsilon?: number
}

export class wBiFPNAdd extends tf.layers.Layer {
    static className  = 'wBiFPNAdd';

    private epsilon: number
    private w: LayerVariable
    // layer weights is named w for consistency with the python implementaiton

    constructor(args: wBiFPNAddArgs) {
        super(args)
        this.epsilon = args.epsilon || 1E-4
    }

    computeOutputShape(inputShape) {
        return tf.gather(inputShape, 0)        
    }

    public build(inputShape: Shape|Shape[]): void {       
        const num_in = inputShape.length
        this.w = this.addWeight(
            this.name,
            [num_in],
            "float32",
            tf.initializers.constant({value: 1 / num_in}),
            null,
            true,
        )
 
    }
   
    // call() is where we do the computation.
    call(input, kwargs) {
        tf.tidy(() => {
            tf.relu
            const result = a.square().log().neg();
            return result;
        });
     }
   
    // Every layer needs a unique name.
    getClassName() { 
        return 'wBiFPNAdd';
    }
}
   