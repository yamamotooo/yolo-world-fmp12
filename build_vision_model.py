"""YOLO-World の Core ML モデルを、FileMaker の Vision 処理で座標が取れる形に変換する。

FileMaker は多次元配列の出力を扱えないため、検出結果を「分類ラベル + 確率」として返す。
  ラベル: "{カード番号}:{軸}:{値}"  例 "0:x1:0.595"  (軸は x1,y1,x2,y2,c。値は 0〜1 を 0.005 刻み)
  確率:   該当するラベルだけ 0.5、それ以外は 0。c は信頼度。
ComputeModel のオプション confidenceLowerLimit を 0.25 などにすると、有効なラベルだけが返る。

使い方: python build_vision_model.py 入力.mlmodel(nms=True で書き出したもの) 出力.mlmodel
"""
import sys

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.proto import FeatureTypes_pb2 as FT
from coremltools.proto import Model_pb2

K_CARDS = 10      # 返すカードの最大数
N = 201           # 値の刻み数 (0.005 刻み)
SHARP = 30.0      # softmax の鋭さ
IOU = 0.45        # NMS の IoU しきい値
SCORE = 0.05      # 検出の信頼度しきい値
AXES = ["x1", "y1", "x2", "y2", "c"]

src = ct.models.MLModel(sys.argv[1]).get_spec()
nn = src.pipeline.models[0]  # image -> var_1183 [8400,4](cx,cy,w,h), var_1179_1 [8400,1]
BOX, SCORES = nn.description.output[1].name, nn.description.output[0].name
NUM_ANCHORS = nn.description.output[1].type.multiArrayType.shape[0]   # 640: 8400, 1280: 33600
assert list(nn.description.output[1].type.multiArrayType.shape) == [NUM_ANCHORS, 4]
IMGSZ = nn.description.input[0].type.imageType.width                  # 書き出し時の入力サイズ

# --- 1. 検出器: 入力を任意サイズにし、先頭で書き出し時の入力サイズにリサイズ ---
b = NeuralNetworkBuilder(spec=nn)
b.add_resize_bilinear("resize_in", "image", "image_resized", target_height=IMGSZ, target_width=IMGSZ,
                      mode="UPSAMPLE_MODE")
layers = list(nn.neuralNetwork.layers)
for l in layers[:-1]:
    for i, n in enumerate(l.input):
        if n == "image":
            l.input[i] = "image_resized"
ordered = [layers[-1]] + layers[:-1]
copies = []
for l in ordered:
    c = type(l)()
    c.CopyFrom(l)
    copies.append(c)
del nn.neuralNetwork.layers[:]
nn.neuralNetwork.layers.extend(copies)
size = nn.description.input[0].type.imageType.imageSizeRange
for r in (size.widthRange, size.heightRange):
    r.lowerBound = 32
    r.upperBound = -1

# --- 2. 分類器: NMS で上位 K 枚を選び、座標を ラベル+確率 に変換 ---
c = NeuralNetworkBuilder([(BOX, datatypes.Array(NUM_ANCHORS, 4)), (SCORES, datatypes.Array(NUM_ANCHORS, 1))],
                         [("probs", datatypes.Array(K_CARDS * len(AXES) * N))],
                         mode="classifier", disable_rank5_shape_mapping=True)
corner = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-.5, 0, .5, 0], [0, -.5, 0, .5]], dtype=np.float64)  # cx,cy,w,h -> x1,y1,x2,y2
c.add_batched_mat_mul("to_corner", [BOX], "corner", weight_matrix_rows=4, weight_matrix_columns=4, W=corner)
c.add_reshape_static("boxes3", "corner", "boxes_b", output_shape=(1, NUM_ANCHORS, 4))
c.add_reshape_static("scores3", SCORES, "scores_b", output_shape=(1, NUM_ANCHORS, 1))
c.add_nms("nms", ["boxes_b", "scores_b"], ["nms_box", "nms_score", "nms_idx", "nms_num"],
          iou_threshold=IOU, score_threshold=SCORE, max_boxes=K_CARDS)
c.add_reshape_static("score_col", "nms_score", "nms_score_c", output_shape=(1, K_CARDS, 1))
c.add_concat_nd("feat", ["nms_box", "nms_score_c"], "feat_o", axis=-1)   # [1,K,5]

W = np.zeros((5, 5 * N))
bias = np.zeros(5 * N)
for a in range(5):
    for k in range(N):
        W[a, a * N + k] = SHARP * 2 * k * (N - 1)
        bias[a * N + k] = -SHARP * k * k
c.add_batched_mat_mul("logit", ["feat_o"], "logits", weight_matrix_rows=5, weight_matrix_columns=5 * N,
                      W=W, bias=bias)
c.add_reshape_static("rs", "logits", "logits4", output_shape=(K_CARDS, 5, N))
c.add_softmax_nd("sm", "logits4", "p4", axis=-1)
c.add_activation("half", "LINEAR", "p4", "p4h", params=[0.5, 0.0])
c.add_reshape_static("mask_in", "nms_score", "sc_k", output_shape=(K_CARDS, 1, 1))
c.add_activation("mask_scale", "LINEAR", "sc_k", "sc_big", params=[1e4, 0.0])
c.add_clip("mask", "sc_big", "valid", min_value=0.0, max_value=1.0)    # 空きスロット(スコア0)は 0
c.add_multiply_broadcastable("apply_mask", ["p4h", "valid"], "p4m")
c.add_reshape_static("flat", "p4m", "probs", output_shape=(K_CARDS * 5 * N,))
labels = [f"{j}:{AXES[a]}:{k / (N - 1):.3f}" for j in range(K_CARDS) for a in range(5) for k in range(N)]
c.set_class_labels(labels, predicted_feature_name="classLabel", prediction_blob="probs")
cls = c.spec
for f in cls.description.input:
    f.type.multiArrayType.dataType = FT.ArrayFeatureType.FLOAT32

# --- 3. パイプライン (分類器として) ---
p = Model_pb2.Model()
p.specificationVersion = 4
p.description.input.add().CopyFrom(nn.description.input[0])
o = p.description.output.add(); o.name = "classLabel"; o.type.stringType.SetInParent()
o = p.description.output.add(); o.name = "probs"; o.type.dictionaryType.stringKeyType.SetInParent()
p.description.predictedFeatureName = "classLabel"
p.description.predictedProbabilitiesName = "probs"
p.pipelineClassifier.pipeline.models.add().CopyFrom(nn)
p.pipelineClassifier.pipeline.models.add().CopyFrom(cls)
ct.models.MLModel(p).save(sys.argv[2])
print("saved:", sys.argv[2])
