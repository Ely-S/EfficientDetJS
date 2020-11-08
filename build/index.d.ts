import { Tensor3D, GraphModel, Tensor4D } from '@tensorflow/tfjs';
interface EfficientDetOptions {
    labels: {};
}
interface Box {
    score: number;
    class: string;
    bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}
export default class EfficientDet {
    labels: {
        [index: number]: string;
    };
    modelURI: string;
    model?: GraphModel;
    minScore: number;
    constructor(options: EfficientDetOptions);
    load: () => Promise<void>;
    predict: (image: Tensor3D) => Promise<Box[]>;
    predictBatch: (imageBatch: Tensor4D) => Promise<Box[]>;
    drawBoxes: (boxes: Box[], canvas: HTMLCanvasElement, clearCanvas?: boolean) => void;
}
export {};
