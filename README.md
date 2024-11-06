# NEW version of this included in yolo repos: [yolov8](https://github.com/lipophrenia/yolov8) and [yolov10](https://github.com/lipophrenia/yolov10)

### `onnx` to `rknn` convertation for Rockchip NPU

Method 1.

In `convert.py` we check the path to the set of jpgs for quantization. You can select a folder with one picture. `dataset.txt` will be generated during execution of the script.
The first argument is the path to the `onnx` model. The second is the target platform (rk3562/rk3566/rk3568/rk3588). The third is the quantization type (i8/fp). The fourth is the `rknn` export path.

```bash
python3 convert.py *.onnx rk3568 fp *.rknn
```

Method 2. Taken from [this](https://github.com/laitathei/YOLOv8-ONNX-RKNN-HORIZON-TensorRT-Segmentation/tree/master),

In `onnx2rknn.py` we check:
- target platform,
- path to a set of jpgs for quantization (similarly, you can specify a directory with one pic),
- model name,
- path to the directory with the trained `onnx` model.

Execute the script.

```bash
python3 onnx2rknn.py
```

If everything is ok, then the model can be used on a Rockchip NPU.
