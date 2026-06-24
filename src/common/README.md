# Utils 工具模块

这个文件夹包含了从 `test_dbscan_crop.py` 中提取的与 agent 逻辑不相关的通用工具函数。

## 模块说明

### 1. `image_utils.py` - 图像处理和编码工具
包含图像处理的基础功能：
- `resize_image_for_sam3()` - 调整图像大小以适配 SAM3 模型
- `resize_mask_for_sam3()` - 调整掩码大小以适配 SAM3 模型
- `encode_image()` - 将 PIL Image 编码为 base64 字符串
- `load_prompt_template()` - 加载系统提示词模板

### 2. `visualization_utils.py` - 可视化工具
包含各种可视化功能：
- `add_axis_labels()` - 为图像添加坐标轴标签
- `draw_points_on_image()` - 在图像上绘制错误点（FP 和 FN）
- `draw_all_points()` - 在图像上绘制所有器官的错误点
- `visualize_clusters()` - 可视化 DBSCAN 聚类结果
- `visualize_bbox()` - 可视化裁剪边界框

### 3. `mask_utils.py` - 掩码处理工具
包含掩码操作的相关功能：
- `ORGANS` - 器官配置常量（颜色、名称等）
- `MASK_ALPHA` - 掩码透明度常量
- `extract_single_organ_mask()` - 从完整掩码中提取单个器官的掩码
- `overlay_single_mask()` - 将单个器官掩码叠加到图像上
- `get_organ_mask_arr()` - 获取单个器官的掩码数组
- `apply_refined_mask()` - 将修复后的掩码应用到完整掩码中
- `overlay_all_masks()` - 将所有器官掩码叠加到图像上

### 4. `clustering_utils.py` - 聚类工具
包含 DBSCAN 聚类相关功能：
- `dbscan_crop()` - 使用 DBSCAN 找到掩码的主要区域并返回裁剪边界框

## 使用方法

```python
from utils.image_utils import encode_image, load_prompt_template
from utils.visualization_utils import add_axis_labels, draw_points_on_image
from utils.mask_utils import ORGANS, extract_single_organ_mask, overlay_single_mask
from utils.clustering_utils import dbscan_crop

# 或者导入所有
from utils import *
```

## 设计原则

这些工具函数被提取出来的原因：
1. **与 agent 逻辑无关** - 这些是纯粹的数据处理和可视化函数
2. **可复用性** - 可以在其他测试或脚本中重用
3. **代码组织** - 将工具函数与业务逻辑分离，提高代码可读性
4. **易于维护** - 集中管理工具函数，便于后续修改和优化

## 保留在 test_dbscan_crop.py 中的函数

以下函数保留在原文件中，因为它们包含 agent 相关的业务逻辑：
- `score_single_organ()` - 包含 LLM 调用和 agent 评分逻辑
- `process_single_image()` - 包含完整的 agent 处理流程
- `test_dbscan_crop()` - 主测试函数

