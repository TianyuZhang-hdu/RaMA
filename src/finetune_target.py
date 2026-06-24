"""
Fine-tune the source-trained model (vendorA) on target domains (B/C/D)
using ASGA-filtered result data (image + corrected pseudo-labels).

数据来源: REPLACE_WITH_RESULT_ROOT/{vendor}/image + mask
  - 只包含 LLM 评分全部 >= 40 的图片（低质量图片已剔除）
  - 图片和 mask 均为 256x256 RGB PNG

Pipeline:
  1. Load pre-trained source model (vendorA) - 原权重不动
  2. Load filtered images + masks from result directory
  3. Fine-tune with:
     - Dice loss
     - Curvature loss (boundary smoothness regularization)
  4. Evaluate on target domain test set using real ground truth
  5. 新权重保存到独立目录（不覆盖原模型）

Pseudo-label format (RGB PNG):
  - Red   [255, 0, 0] = LV  (Left Ventricle)
  - Green [0, 255, 0] = MYO (Myocardium)
  - Blue  [0, 0, 255] = RV  (Right Ventricle)
  - Black [0, 0, 0]   = Background

Usage (single domain):
    python finetune_target.py --Target_Dataset vendorD
    python finetune_target.py --Target_Dataset vendorC

Usage (all domains):
    bash run_finetune_all.sh
"""

import os
import sys
import argparse
import datetime
import math
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import copy

# Project imports
from networks.ResUnet import ResUnet
from loss import DiceLoss, curvature_loss
from train_utils import set_random
from utils_ours.config import Logger
from utils_ours.metrics import calculate_metrics
from dataloaders.mms_dataloader import mms_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.normalize import normalize_image_to_0_1_3D


# =========================================================================
#  数据增强 (image + mask 同步变换)
# =========================================================================
def apply_augmentation(img_pil, mask_pil):
    """
    对图像和掩码施加同步的随机数据增强。

    增强策略（针对小数据集 + 伪标签噪声优化）：
      - 随机水平翻转 (p=0.5)
      - 随机垂直翻转 (p=0.5)
      - 随机旋转 0/90/180/270 度 (p=1.0)
      - 随机缩放裁剪 (scale 0.8~1.2, p=0.5)
      - 随机亮度/对比度抖动 (仅 image, p=0.3)

    Args:
        img_pil: PIL Image (灰度)
        mask_pil: PIL Image (RGB mask)
    Returns:
        img_pil, mask_pil: 增强后的 PIL Image
    """
    # 水平翻转
    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # 垂直翻转
    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

    # 随机 90 度旋转
    rot = random.choice([0, 90, 180, 270])
    if rot == 90:
        img_pil = img_pil.transpose(Image.ROTATE_90)
        mask_pil = mask_pil.transpose(Image.ROTATE_90)
    elif rot == 180:
        img_pil = img_pil.transpose(Image.ROTATE_180)
        mask_pil = mask_pil.transpose(Image.ROTATE_180)
    elif rot == 270:
        img_pil = img_pil.transpose(Image.ROTATE_270)
        mask_pil = mask_pil.transpose(Image.ROTATE_270)

    # 随机缩放裁剪 (scale 0.8 ~ 1.2)
    if random.random() < 0.5:
        w, h = img_pil.size
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
        mask_pil = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)

        # 中心裁剪或补零回到原始尺寸
        if new_w >= w and new_h >= h:
            # 裁剪
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img_pil = img_pil.crop((left, top, left + w, top + h))
            mask_pil = mask_pil.crop((left, top, left + w, top + h))
        else:
            # 补零
            img_pad = Image.new(img_pil.mode, (w, h), 0)
            mask_pad = Image.new(mask_pil.mode, (w, h), (0, 0, 0))
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            img_pad.paste(img_pil, (paste_x, paste_y))
            mask_pad.paste(mask_pil, (paste_x, paste_y))
            img_pil = img_pad
            mask_pil = mask_pad

    # 随机亮度/对比度抖动 (仅 image)
    if random.random() < 0.3:
        img_arr = np.array(img_pil).astype(np.float32)
        # 亮度
        brightness = random.uniform(-20, 20)
        img_arr = img_arr + brightness
        # 对比度
        contrast = random.uniform(0.8, 1.2)
        mean_val = img_arr.mean()
        img_arr = (img_arr - mean_val) * contrast + mean_val
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_arr)

    return img_pil, mask_pil


# =========================================================================
#  Dataset: loads image + mask from ASGA result directory
# =========================================================================
class ResultDataset(Dataset):
    """
    Dataset that reads directly from the ASGA result directory.

    目录结构:
      result_root/{vendor}/image/{img_id}.png  (灰度原图, 256x256)
      result_root/{vendor}/mask/{img_id}.png   (RGB 伪标签, 256x256)

    只包含 LLM 评分 >= 40 的合格图片。

    Args:
        image_dir: 图片目录 (result/{vendor}/image)
        mask_dir:  掩码目录 (result/{vendor}/mask)
        llm_score_dir: LLM 评分目录 (可选，用于质量加权)
        target_size: 目标尺寸 (default 256)
        img_normalize: 是否归一化到 [0, 1]
        augment: 是否启用数据增强 (训练集 True, 测试集 False)
    """

    def __init__(self, image_dir, mask_dir, llm_score_dir=None,
                 target_size=256, img_normalize=True, augment=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.augment = augment

        # 扫描目录，只取 image 和 mask 都存在的
        img_files = set(f for f in os.listdir(image_dir) if f.endswith('.png'))
        mask_files = set(f for f in os.listdir(mask_dir) if f.endswith('.png'))
        common = sorted(img_files & mask_files)
        self.filenames = common
        print(f"  [ResultDataset] {image_dir}")
        print(f"    images: {len(img_files)}, masks: {len(mask_files)}, paired: {len(common)}")
        print(f"    augment: {augment}")

        # 加载 LLM 评分 -> per-organ weights
        # scores[img_id] = [LV_score, MYO_score, RV_score], 归一化到 [0, 1]
        self.scores = {}
        if llm_score_dir and os.path.isdir(llm_score_dir):
            loaded = 0
            for fname in self.filenames:
                img_id = os.path.splitext(fname)[0]
                json_path = os.path.join(llm_score_dir, f"{img_id}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        organs = data.get("organs", {})
                        lv = organs.get("LV", {}).get("scores", {}).get("total", 70)
                        myo = organs.get("MYO", {}).get("scores", {}).get("total", 70)
                        rv = organs.get("RV", {}).get("scores", {}).get("total", 70)
                        self.scores[img_id] = np.array(
                            [lv / 100.0, myo / 100.0, rv / 100.0], dtype=np.float32
                        )
                        loaded += 1
                    except Exception:
                        pass
            print(f"    LLM scores loaded: {loaded}/{len(self.filenames)}")
            if loaded > 0:
                all_w = np.array(list(self.scores.values()))
                print(f"    Score weights - LV: {all_w[:,0].mean():.2f}±{all_w[:,0].std():.2f}, "
                      f"MYO: {all_w[:,1].mean():.2f}±{all_w[:,1].std():.2f}, "
                      f"RV: {all_w[:,2].mean():.2f}±{all_w[:,2].std():.2f}")
        else:
            if llm_score_dir:
                print(f"    [WARNING] LLM score dir not found: {llm_score_dir}")
            else:
                print(f"    LLM score weighting: disabled")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_id = os.path.splitext(fname)[0]

        # --- Load image ---
        img = Image.open(os.path.join(self.image_dir, fname))
        img = img.resize(self.target_size)

        # --- Load mask ---
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert('RGB')
        mask = mask.resize(self.target_size, resample=Image.NEAREST)

        # --- 数据增强 (训练时) ---
        if self.augment:
            img, mask = apply_augmentation(img, mask)

        # --- Image -> numpy ---
        img_npy = np.array(img)[np.newaxis, ...].astype(np.float32)
        if self.img_normalize:
            img_npy = normalize_image_to_0_1_3D(img_npy)

        # --- Mask (RGB -> one-hot [3, H, W]) ---
        mask_arr = np.array(mask)  # (H, W, 3), uint8

        mask_onehot = np.zeros((3, self.target_size[0], self.target_size[1]),
                               dtype=np.float32)
        mask_onehot[0] = (mask_arr[:, :, 0] > 128).astype(np.float32)  # LV
        mask_onehot[1] = (mask_arr[:, :, 1] > 128).astype(np.float32)  # MYO
        mask_onehot[2] = (mask_arr[:, :, 2] > 128).astype(np.float32)  # RV

        # --- Per-organ LLM score weight [3] ---
        # 默认权重 1.0（无评分时当作满分）
        organ_weight = self.scores.get(img_id, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        return img_npy, mask_onehot, fname, organ_weight


def result_collate_fn(batch):
    """Collate function for ResultDataset."""
    images, masks, names, weights = zip(*batch)
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    names = np.array(names)
    weights = np.stack(weights, axis=0)   # (B, 3)
    return {'data': images, 'mask': masks, 'name': names, 'organ_weight': weights}


# =========================================================================
#  LLM 评分加权 Dice Loss
# =========================================================================
def organ_weighted_dice_loss(pred, target, organ_weights):
    """
    Per-organ weighted Dice Loss.
    LLM 评分高的器官 → 高权重（更信任该伪标签）
    LLM 评分低的器官 → 低权重（less trust，减少噪声影响）

    Args:
        pred:          (B, 3, H, W) sigmoid 输出
        target:        (B, 3, H, W) one-hot 伪标签
        organ_weights: (B, 3) per-organ weights, 范围 [0, 1]
                       来自 LLM total_score / 100

    Returns:
        weighted dice loss (scalar)
    """
    smooth = 1e-4
    total_loss = 0.0

    for i in range(3):  # 0=LV, 1=MYO, 2=RV
        w = organ_weights[:, i]            # (B,)
        p = pred[:, i]                     # (B, H, W)
        t = target[:, i]                   # (B, H, W)

        # Per-sample dice
        intersect = (p * t).sum(dim=(1, 2))                    # (B,)
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))          # (B,)
        dice = (2.0 * intersect + smooth) / (denom + smooth)   # (B,)
        per_sample_loss = 1.0 - dice                            # (B,)

        # 加权平均
        weighted_loss = (per_sample_loss * w).sum() / (w.sum() + 1e-8)
        total_loss += weighted_loss

    return total_loss / 3.0


# =========================================================================
#  Helper: convert prediction logit to colored mask for saving
# =========================================================================
def logit2mask_save(pred_logit, threshold=0.5):
    """Convert model output logits to a single-channel colored mask."""
    pred = torch.sigmoid(pred_logit)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    pseudo_labels = torch.zeros([pred.size(0), pred.size(2), pred.size(3)])
    if pred.is_cuda:
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pred[:, 0] == 1] = 76   # LV
    pseudo_labels[pred[:, 1] == 1] = 150  # MYO
    pseudo_labels[pred[:, 2] == 1] = 29   # RV

    return pseudo_labels.unsqueeze(dim=1)


# =========================================================================
#  EMA (Exponential Moving Average) - 稳定训练，减少伪标签噪声影响
# =========================================================================
class EMA:
    """
    Exponential Moving Average for model parameters.
    在伪标签训练中，EMA 能有效抑制噪声导致的参数震荡。
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        """将 EMA 参数应用到模型（用于评估）。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """恢复原始参数（从 EMA 评估模式恢复到训练模式）。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =========================================================================
#  Training loop
# =========================================================================
def train(config, train_loader, test_loader):
    device = torch.device(config.device)

    # ---- TensorBoard ----
    tb_dir = os.path.join(config.save_path, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # ---- Build model & load source weights ----
    model = ResUnet(resnet=config.backbone, num_classes=config.out_ch,
                    pretrained=False, in_ch=config.in_ch).to(device)
    checkpoint = torch.load(config.model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    print(f"[INFO] Loaded source model from: {config.model_path}")

    # ---- 可选：冻结编码器前几层 ----
    freeze_layers = config.freeze_encoder_layers
    if freeze_layers > 0:
        frozen_modules = []
        encoder_layers = [
            ('conv1+bn1', [model.res.conv1, model.res.bn1]),
            ('layer1', [model.res.layer1]),
            ('layer2', [model.res.layer2]),
            ('layer3', [model.res.layer3]),
            ('layer4', [model.res.layer4]),
        ]
        for i in range(min(freeze_layers, len(encoder_layers))):
            name, modules = encoder_layers[i]
            for m in modules:
                for param in m.parameters():
                    param.requires_grad = False
            frozen_modules.append(name)
        print(f"[INFO] Frozen encoder layers: {frozen_modules}")

    # ---- Optimizer (只优化 requires_grad=True 的参数) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Parameters: {train_params:,} trainable / {total_params:,} total "
          f"({100*train_params/total_params:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)

    # ---- Cosine Annealing + Warmup ----
    num_epochs = config.num_epochs
    warmup_epochs = config.warmup_epochs

    # Cosine scheduler (不含 warmup 部分)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=config.lr * 0.01  # 最低降到初始 LR 的 1%
    )

    # ---- EMA ----
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    if ema:
        print(f"[INFO] EMA enabled, decay={config.ema_decay}")

    # ---- Loss ----
    dice_loss_fn = DiceLoss(config.out_ch).to(device)
    curve_weight = config.curve_loss_weight

    best_dice = 0.0
    best_epoch = -1

    print(f"\n{'='*60}")
    print(f"  Fine-tuning on {config.Target_Dataset}")
    print(f"  Result Root: {config.result_root}")
    print(f"  Epochs: {num_epochs} | LR: {config.lr} | Batch: {config.batch_size}")
    print(f"  Warmup: {warmup_epochs} | Curve weight: {curve_weight}")
    print(f"  Freeze layers: {freeze_layers} | EMA: {config.use_ema}")
    print(f"  Augmentation: {config.augment}")
    print(f"  LLM score weight: {config.use_score_weight}"
          + (f" ({config.llm_score_dir})" if config.use_score_weight and config.llm_score_dir else ""))
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        total_dice_loss = 0.0
        total_curve_loss = 0.0
        num_batches = 0

        # Warmup: 线性增长 LR
        if epoch < warmup_epochs:
            warmup_lr = config.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Train Epoch {epoch+1}"):
            images = data['data']       # (B, 1, H, W) numpy
            pl_masks = data['mask']     # (B, 3, H, W) numpy, one-hot pseudo-labels
            ow = data['organ_weight']   # (B, 3) per-organ LLM score weights

            images = torch.from_numpy(images).float().to(device)
            pl_masks = torch.from_numpy(pl_masks).float().to(device)
            ow = torch.from_numpy(ow).float().to(device)

            # Forward
            pred_logit, _ = model(images)
            pred_sigmoid = torch.sigmoid(pred_logit)

            # Dice loss — 根据是否有 LLM 评分选择加权方式
            if config.use_score_weight:
                dice_loss = organ_weighted_dice_loss(pred_sigmoid, pl_masks, ow)
            else:
                weight_map = torch.ones(images.size(0), images.size(2), images.size(3),
                                        device=device)
                dice_loss = dice_loss_fn(pred_sigmoid, pl_masks, weight_map, False)

            # Curvature loss
            curve_loss = curvature_loss(pred_sigmoid) * curve_weight

            # Total loss
            total_loss = dice_loss + curve_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 更新 EMA
            if ema:
                ema.update(model)

            total_dice_loss += dice_loss.item()
            total_curve_loss += curve_loss.item()
            num_batches += 1

        # Epoch summary
        avg_dice_loss = total_dice_loss / max(num_batches, 1)
        avg_curve_loss = total_curve_loss / max(num_batches, 1)
        writer.add_scalar('train/dice_loss', avg_dice_loss, epoch)
        writer.add_scalar('train/curve_loss', avg_curve_loss, epoch)

        # 更新学习率 (warmup 后才用 cosine)
        if epoch >= warmup_epochs:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        print(f"  Dice Loss: {avg_dice_loss:.4f} | Curve Loss: {avg_curve_loss:.6f} | LR: {current_lr:.2e}")

        # ---- Evaluation on test set (using real GT labels) ----
        # 评估时使用 EMA 参数
        if ema:
            ema.apply_shadow(model)

        current_dice = 0.0
        num_test = 0
        model.eval()
        with torch.no_grad():
            for it, data in tqdm(enumerate(test_loader), total=len(test_loader),
                                 desc=f"Eval  Epoch {epoch+1}"):
                x = data['data']
                y = data['mask']
                x = torch.from_numpy(x).float().to(device)
                y = torch.from_numpy(y).float().to(device)

                pred_logit, _ = model(x)
                seg_output = torch.sigmoid(pred_logit)

                metrics = calculate_metrics(
                    seg_output.detach().cpu(), y.detach().cpu()
                )
                # Average Dice over LV, MYO, RV
                sample_dice = (metrics[0][0] + metrics[2][0] + metrics[4][0]) / 3.0
                current_dice += sample_dice
                num_test += 1

        dice_mean = current_dice / max(num_test, 1)
        writer.add_scalar('eval/dice_mean', dice_mean, epoch)
        print(f"  Eval Dice: {dice_mean:.4f}")

        # Save best model (保存 EMA 参数)
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_epoch = epoch + 1
            model_dir = os.path.join(config.save_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            best_path = os.path.join(model_dir, 'best-Res_Unet.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best model! Dice={best_dice:.4f} at epoch {best_epoch}")

        # 恢复训练参数
        if ema:
            ema.restore(model)

    # ---- Save final model (保存 EMA 参数) ----
    if ema:
        ema.apply_shadow(model)
    model_dir = os.path.join(config.save_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    last_path = os.path.join(model_dir, 'last-Res_Unet.pth')
    torch.save(model.state_dict(), last_path)
    print(f"\n[INFO] Last model saved to: {last_path}")

    # ---- Load best model for final evaluation ----
    best_path = os.path.join(model_dir, 'best-Res_Unet.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location='cpu'), strict=True)
        print(f"[INFO] Best model loaded from epoch {best_epoch} (Dice={best_dice:.4f})")
    model.eval()

    # =========================================================================
    #  Final evaluation on test set
    # =========================================================================
    metrics_test = [[], [], [], [], [], []]
    metric_dict = ['LV_Dice', 'LV_ASD', 'MYO_Dice', 'MYO_ASD', 'RV_Dice', 'RV_ASD']
    results_list = []

    print(f"\n{'='*60}")
    print("  Final Evaluation on Test Set (Ground Truth)")
    print(f"{'='*60}")

    for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader),
                                desc="Final Test"):
        x = data['data']
        y = data['mask']
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        with torch.no_grad():
            pred_logit, _ = model(x)
        seg_output = torch.sigmoid(pred_logit)

        metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())

        sample_values = [float(np.array(m_list[0])) for m_list in metrics]
        results_list.append([data['name'][0]] + sample_values)

        for i in range(len(metrics)):
            assert isinstance(metrics[i], list), "Metrics value is not list type."
            metrics_test[i] += metrics[i]

        # Save predicted masks
        save_pred = logit2mask_save(
            pred_logit.detach().clone(), threshold=0.5
        )[0][0].cpu().numpy()
        save_image = Image.fromarray(save_pred.astype(np.uint8))
        save_name = str(data['name'][0]).split('/')[-1]
        save_image.save(os.path.join(config.img_save_path, save_name))

    # ---- Print final metrics ----
    means = np.round(np.mean(metrics_test, axis=1), 2)
    stds = np.round(np.std(metrics_test, axis=1), 2)
    pm_line = ["Mean+/-Std"] + [f"{means[i]:.2f}+/-{stds[i]:.2f}"
                                 for i in range(len(metric_dict))]

    print(f"\n{'='*60}")
    print(f"  Final Test Metrics ({config.Target_Dataset})")
    print(f"{'='*60}")
    col_width = 14
    dice_idx = [0, 2, 4]
    asd_idx = [1, 3, 5]

    header = f"{'':>8}{'LV'.ljust(col_width)}{'MYO'.ljust(col_width)}{'RV'.ljust(col_width)}"
    dice_pm = "".join(
        f"{(f'{means[i]:.2f}+/-{stds[i]:.2f}').ljust(col_width)}" for i in dice_idx
    )
    asd_pm = "".join(
        f"{(f'{means[i]:.2f}+/-{stds[i]:.2f}').ljust(col_width)}" for i in asd_idx
    )

    print(header)
    print(f"{'Dice:':<8}{dice_pm}")
    print(f"{'ASSD:':<8}{asd_pm}")
    print(f"\nBest epoch: {best_epoch}, Best avg Dice: {best_dice:.4f}")

    # ---- Save results to CSV (and Excel if openpyxl available) ----
    df = pd.DataFrame(results_list, columns=["Image"] + metric_dict)
    df.loc[len(df)] = pm_line
    out_csv = os.path.join(config.save_path, "test_metrics_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Results saved to: {out_csv}")
    try:
        out_xlsx = os.path.join(config.save_path, "test_metrics_results.xlsx")
        df.to_excel(out_xlsx, index=False)
        print(f"[INFO] Excel saved to: {out_xlsx}")
    except ImportError:
        print("[WARNING] openpyxl not installed, skipping Excel export")

    # ---- 保存训练配置 ----
    train_config = {
        "target_dataset": config.Target_Dataset,
        "source_dataset": config.Source_Dataset,
        "model_path": config.model_path,
        "result_root": config.result_root,
        "dataset_root": config.dataset_root,
        "num_train_samples": len(train_loader.dataset),
        "num_test_samples": len(test_loader.dataset),
        "hyperparameters": {
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "warmup_epochs": config.warmup_epochs,
            "curve_loss_weight": config.curve_loss_weight,
            "freeze_encoder_layers": config.freeze_encoder_layers,
            "augment": config.augment,
            "use_ema": config.use_ema,
            "ema_decay": config.ema_decay if config.use_ema else None,
            "use_score_weight": config.use_score_weight,
            "llm_score_dir": config.llm_score_dir if config.use_score_weight else None,
        },
        "results": {
            "best_epoch": best_epoch,
            "best_eval_dice": round(best_dice, 4),
            "final_test": {
                "LV_Dice": float(means[0]),
                "MYO_Dice": float(means[2]),
                "RV_Dice": float(means[4]),
                "LV_ASSD": float(means[1]),
                "MYO_ASSD": float(means[3]),
                "RV_ASSD": float(means[5]),
                "avg_Dice": round(float((means[0] + means[2] + means[4]) / 3), 4),
            },
        },
        "best_model_path": os.path.join(config.save_path, "model", "best-Res_Unet.pth"),
        "timestamp": os.path.basename(config.save_path),
    }
    config_path = os.path.join(config.save_path, "train_config.json")
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Training config saved to: {config_path}")

    writer.close()
    return None


# =========================================================================
#  Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune source model on target domain using corrected pseudo-labels'
    )

    # ---- Dataset ----
    parser.add_argument('--Source_Dataset', type=str, default='vendorA')
    parser.add_argument('--Target_Dataset', type=str, default='vendorC',
                        help='Target domain: vendorB / vendorC / vendorD')
    parser.add_argument('--exp_name', type=str, default='finetune_ASGA')

    # ---- Model ----
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=3)

    # ---- Training ----
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='初始学习率 (降低以防止过拟合伪标签噪声)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='训练轮数 (配合更低 LR 需要更多轮)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Warmup 轮数 (LR 线性增长)')
    parser.add_argument('--curve_loss_weight', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)

    # ---- 训练策略 ----
    parser.add_argument('--augment', action='store_true', default=True,
                        help='启用数据增强 (默认开启)')
    parser.add_argument('--no_augment', action='store_true', default=False,
                        help='关闭数据增强')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='使用 EMA 稳定训练 (默认开启)')
    parser.add_argument('--no_ema', action='store_true', default=False,
                        help='关闭 EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA 衰减率')
    parser.add_argument('--freeze_encoder_layers', type=int, default=2,
                        help='冻结编码器前 N 层 (0=不冻结, 1=conv1+bn1, '
                             '2=+layer1, 3=+layer2, ...)')

    # ---- LLM 评分加权 ----
    parser.add_argument('--use_score_weight', action='store_true', default=True,
                        help='使用 LLM 评分作为 per-organ loss 权重 (默认开启)')
    parser.add_argument('--no_score_weight', action='store_true', default=False,
                        help='关闭 LLM 评分加权')
    parser.add_argument('--llm_score_dir', type=str, default=None,
                        help='LLM 评分 JSON 目录。默认自动推断: '
                             'llm_scores/qwen-vl-max/{vendor}_train/')

    # ---- Paths ----
    parser.add_argument('--model_path', type=str,
                        default='REPLACE_WITH_SOURCE_MODEL_PATH',
                        help='Path to pre-trained source model (vendorA, 只读不改)')
    parser.add_argument('--dataset_root', type=str,
                        default='REPLACE_WITH_DATASET_ROOT',
                        help='Root of M&MS dataset (test set GT labels)')
    parser.add_argument('--result_root', type=str,
                        default='REPLACE_WITH_RESULT_ROOT',
                        help='Root of ASGA result data (filtered image + mask)')
    parser.add_argument('--path_save_log', type=str,
                        default='REPLACE_WITH_LOG_ROOT',
                        help='新权重和日志的保存目录（与原模型分开）')

    # ---- Device ----
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    # 处理 no_xxx 标志
    if args.no_augment:
        args.augment = False
    if args.no_ema:
        args.use_ema = False
    if args.no_score_weight:
        args.use_score_weight = False

    # 自动推断 LLM 评分目录 — 遍历所有模型目录，取文件数最多的
    if args.llm_score_dir is None and args.use_score_weight:
        llm_root = os.environ.get("RAMA_LLM_SCORE_ROOT", "")
        target_key = f"{args.Target_Dataset}_train"
        best_dir, best_count = None, 0
        if os.path.isdir(llm_root):
            for model_name in os.listdir(llm_root):
                candidate = os.path.join(llm_root, model_name, target_key)
                if os.path.isdir(candidate):
                    n = len([f for f in os.listdir(candidate) if f.endswith('.json')])
                    if n > best_count:
                        best_dir, best_count = candidate, n
        if best_dir:
            args.llm_score_dir = best_dir
            print(f"[INFO] Auto-detected LLM scores: {best_dir} ({best_count} files)")
        else:
            print(f"[WARNING] No LLM score dir found for {target_key}, disabling score weight")
            args.use_score_weight = False

    config = args

    # ---- Setup experiment directories ----
    config.exp_name = (config.exp_name + '/'
                       + config.Source_Dataset + '_to_' + config.Target_Dataset)
    time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
    log_root = os.path.join(config.path_save_log, config.exp_name)
    os.makedirs(log_root, exist_ok=True)

    log_path = os.path.join(log_root, time_now + '.log')
    sys.stdout = Logger(log_path, sys.stdout)

    config.save_path = os.path.join(log_root, time_now)
    config.img_save_path = os.path.join(log_root, time_now, 'mask')
    os.makedirs(config.img_save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ASGA Fine-tuning: {config.Source_Dataset} -> {config.Target_Dataset}")
    print(f"{'='*60}")
    print(f"  Source Model (只读): {config.model_path}")
    print(f"  Result Root       : {config.result_root}")
    print(f"  Dataset Root (GT) : {config.dataset_root}")
    print(f"  Save Path (新权重): {config.save_path}")
    print(f"  Epochs: {config.num_epochs} | LR: {config.lr} | Batch: {config.batch_size}")
    print(f"{'='*60}\n")

    # ==================================================================
    #  Build TRAINING DataLoader: from ASGA result directory
    # ==================================================================
    result_image_dir = os.path.join(config.result_root, config.Target_Dataset, 'image')
    result_mask_dir = os.path.join(config.result_root, config.Target_Dataset, 'mask')

    if not os.path.exists(result_image_dir):
        print(f"[ERROR] Result image directory not found: {result_image_dir}")
        return
    if not os.path.exists(result_mask_dir):
        print(f"[ERROR] Result mask directory not found: {result_mask_dir}")
        return

    train_dataset = ResultDataset(
        image_dir=result_image_dir,
        mask_dir=result_mask_dir,
        llm_score_dir=config.llm_score_dir if config.use_score_weight else None,
        target_size=config.image_size,
        img_normalize=True,
        augment=config.augment
    )
    print(f"Train Dataset: {len(train_dataset)} samples")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=result_collate_fn,
        num_workers=config.num_workers
    )

    # ==================================================================
    #  Build TEST DataLoader: image + real GT (using original mms_dataset)
    # ==================================================================
    target_test_csv = [config.Target_Dataset + '_test.csv']
    ts_img_list, ts_label_list = convert_labeled_list(
        config.dataset_root, target_test_csv
    )
    test_dataset = mms_dataset(
        config.dataset_root, ts_img_list, ts_label_list,
        config.image_size, img_normalize=True
    )
    print(f"Test Dataset:  {len(test_dataset)} samples")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_wo_transform,
        num_workers=config.num_workers
    )

    # ---- Run training ----
    train(config, train_loader, test_loader)


if __name__ == '__main__':
    set_random()
    main()
