"""
工具函数模块
用于存放与 agent 逻辑无关的通用工具函数
"""

from .image_utils import (
    resize_image_for_sam3,
    resize_mask_for_sam3,
    encode_image,
    load_prompt_template
)

from .visualization_utils import (
    add_axis_labels,
    draw_points_on_image,
    draw_all_points,
    visualize_clusters,
    visualize_bbox
)

from .mask_utils import (
    ORGANS, MASK_ALPHA,
    extract_single_organ_mask, overlay_single_mask, 
    get_organ_mask_arr, apply_refined_mask, overlay_all_masks,
    compute_dice, compute_all_dice,
    compute_assd, compute_all_assd, compute_all_metrics
)

from .clustering_utils import (
    dbscan_crop,
    get_organ_bbox,
    get_all_organ_bboxes
)

from .json_utils import (
    extract_first_json,
    parse_llm_response
)

from .plot_utils import (
    plot_debug_results,
    save_final_mask
)

__all__ = [
    # image_utils
    'resize_image_for_sam3',
    'resize_mask_for_sam3',
    'encode_image',
    'load_prompt_template',
    # visualization_utils
    'add_axis_labels',
    'draw_points_on_image',
    'draw_all_points',
    'visualize_clusters',
    'visualize_bbox',
    # mask_utils
    'ORGANS',
    'MASK_ALPHA',
    'extract_single_organ_mask',
    'overlay_single_mask',
    'get_organ_mask_arr',
    'apply_refined_mask',
    'overlay_all_masks',
    'compute_dice',
    'compute_all_dice',
    'compute_assd',
    'compute_all_assd',
    'compute_all_metrics',
    # clustering_utils
    'dbscan_crop',
    'get_organ_bbox',
    'get_all_organ_bboxes',
    # json_utils
    'extract_first_json',
    'parse_llm_response',
    # plot_utils
    'plot_debug_results',
    'save_final_mask',
]
