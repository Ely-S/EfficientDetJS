"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs");
// Labels for the COCO dataset
var labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
};
var EfficientDet = /** @class */ (function () {
    function EfficientDet(options) {
        var _this = this;
        this.labels = labels;
        this.modelURI = 'd0/model.json';
        this.minScore = 0.2;
        this.load = function () { return __awaiter(_this, void 0, void 0, function () {
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0: return [4 /*yield*/, tf.ready()];
                    case 1:
                        _b.sent();
                        _a = this;
                        return [4 /*yield*/, tf.loadGraphModel(this.modelURI)];
                    case 2:
                        _a.model = _b.sent();
                        return [2 /*return*/];
                }
            });
        }); };
        this.predict = function (image) { return __awaiter(_this, void 0, void 0, function () {
            var batch;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        batch = image.expandDims();
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, , 3, 4]);
                        return [4 /*yield*/, this.predictBatch(batch)];
                    case 2: return [2 /*return*/, _a.sent()];
                    case 3:
                        tf.dispose(batch);
                        return [7 /*endfinally*/];
                    case 4: return [2 /*return*/];
                }
            });
        }); };
        this.predictBatch = function (imageBatch) { return __awaiter(_this, void 0, void 0, function () {
            var result, detectedObjects, results, predictions;
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.model) {
                            throw new Error("Model not loaded yet, call .load()");
                        }
                        return [4 /*yield*/, this.model.executeAsync({ image_arrays: imageBatch })];
                    case 1:
                        result = _a.sent();
                        detectedObjects = [];
                        return [4 /*yield*/, result.array()];
                    case 2:
                        results = _a.sent();
                        predictions = results[0];
                        predictions.forEach(function (out) {
                            var image_id = out[0], y = out[1], x = out[2], ymax = out[3], xmax = out[4], score = out[5], _class = out[6];
                            if (score < _this.minScore)
                                return;
                            detectedObjects.push({
                                bbox: {
                                    x: x,
                                    y: y,
                                    width: xmax - x,
                                    height: ymax - y
                                },
                                "class": _this.labels[_class],
                                score: score
                            });
                        });
                        tf.dispose(result);
                        return [2 /*return*/, detectedObjects];
                }
            });
        }); };
        this.drawBoxes = function (boxes, canvas, clearCanvas) {
            if (clearCanvas === void 0) { clearCanvas = false; }
            var ctx = canvas.getContext("2d");
            if (clearCanvas)
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            var font = "12px ariel";
            ctx.font = font;
            ctx.textBaseline = "top";
            boxes.forEach(function (prediction) {
                // let [x, y, width, height] = prediction.bbox
                var _a = prediction.bbox, x = _a.x, y = _a.y, width = _a.width, height = _a.height;
                // Draw the bounding box.
                ctx.strokeStyle = "#FF0000";
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, width, height);
                // Draw the label background.
                var label = (prediction.score * 100).toFixed(0);
                ctx.fillStyle = "#FF0000";
                var textWidth = ctx.measureText(label).width * 1.3;
                var textHeight = parseInt(font, 10);
                +2;
                // draw top left rectangle
                ctx.fillRect(x, y - 2, textWidth, textHeight);
                // draw bottom left rectangle
                ctx.fillRect(x, y + height - textHeight, textWidth, textHeight);
                // Draw the text last to ensure  it's on top.
                ctx.fillStyle = "#ffff";
                ctx.fillText(prediction.class, x, y);
                ctx.fillText(label, x, y + height - textHeight);
            });
        };
    }
    return EfficientDet;
}());
exports.default = EfficientDet;
