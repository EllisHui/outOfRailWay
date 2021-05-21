- **2021.04.26**: Above problem solved by replace F.conv2d by matmul. Now it passes this step. Now facing new issue:

  ```
  While parsing node number 600 [Gather -> "1786"]:
  ERROR: /home/onnx-tensorrt/builtin_op_importers.cpp:1147 In function importGather:
[8] Assertion failed: data.getType() != nvinfer1::DataType::kBOOL
  
```
  
  Which caused by this line code:
  
  ```
  # sort and keep top nms_pre
  _, sort_inds = torch.sort(cate_scores, descending=True)
  if len(sort_inds) > self.max_before_nms:
  sort_inds = sort_inds[:self.max_before_nms]
  seg_masks = seg_masks[sort_inds, :, :]
  ```
  
  
  
- **2021.02.20**: Got some error messages:

  ```
  [8] Assertion failed: ctx->network()->hasExplicitPrecision() && "TensorRT only supports multi-input conv for explicit precision QAT networks!"
  ```

  indicates me that model have explicit data type, but this is not QAT models. I fired an issue report this: https://github.com/onnx/onnx-tensorrt/issues/645

  I hope there will be a response of this error. Currently, I have no solution for this.

