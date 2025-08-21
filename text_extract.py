# -*- coding: utf-8 -*-
"""
CLIP 文本向量提取 + Prompt 集成降相似度 + （可选）PCA Whitening
+ 相关度矩阵（含数值标注）+ 打印并保存每类最终 prompts

依赖: torch, transformers, numpy, matplotlib (Python 3.8+)
可选: scikit-learn (仅用于 PCA 2D 可视化；未安装会自动跳过)
"""

import os
import json
import math
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# ------------------------- 可调参数 -------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "outputs"
SAVE_EMB_PATH = os.path.join(SAVE_DIR, "drone_text_embeddings.pt")
SAVE_SIM_NPY = os.path.join(SAVE_DIR, "drone_text_similarity.npy")
SAVE_SIM_CSV = os.path.join(SAVE_DIR, "drone_text_similarity.csv")
SAVE_FIG_PATH = os.path.join(SAVE_DIR, "drone_text_similarity.png")
SAVE_PCA_PATH = os.path.join(SAVE_DIR, "pca_scatter.png")
SAVE_PROMPTS_TXT = os.path.join(SAVE_DIR, "resolved_prompts.txt")
SAVE_PROMPTS_JSON = os.path.join(SAVE_DIR, "resolved_prompts.json")

# Prompt 集成（多模板平均）
USE_PROMPT_ENSEMBLE = True

# 语义去相关（PCA Whitening）
USE_WHITENING = True
WHITENING_DIM: Optional[int] = None  # None=全维白化；也可设 256/384 等

# 英文/中文模板是否启用
USE_EN_PROMPTS = True
USE_ZH_PROMPTS = True

# 自动注入“类别差异化描述”（品牌/系列/频段/信号形态等）
USE_AUTO_DESCRIPTORS = True

# 热力图是否在每个格子上标数值（大矩阵时绘制会慢）
ANNOTATE_VALUES = True
# -----------------------------------------------------------

# 原始类别
drone_categories = [
    "background_noise",
    "DJI Phantom 3",
    "DJI Phantom 4 Pro",
    "DJI MATRICE 200",
    "DJI MATRICE 100",
    "DJI Air 2S",
    "DJI Mini 3 Pro",
    "DJI Inspire 2",
    "DJI Mavic Pro",
    "DJI Mini 2",
    "DJI Mavic 3",
    "DJI MATRICE 300",
    "DJI Phantom 4 Pro RTK",
    "DJI MATRICE 30T",
    "DJI AVATA",
    "DJI 通信模块自组机",
    "DJI MATRICE 600 Pro",
    "VBar 飞控器",
    "FrSky X20 飞控器",
    "Futaba T6IZ 飞控器",
    "Taranis Plus 飞控器",
    "RadioLink AT9S 飞控器",
    "Futaba T14SG 飞控器",
    "云卓 T12 飞控器"
]

def auto_descriptor(name: str):
    """根据名称自动生成区分度更高的中英文描述片段。"""
    name_l = name.lower()
    brand = ""
    family = ""
    type_hint = "RF flight-control signal spectrogram"
    band_hint = ""
    extra = []

    if "dji" in name_l:
        brand = "DJI"
        if "phantom" in name_l: family = "Phantom series"
        elif "mavic" in name_l: family = "Mavic series"
        elif "matrice" in name_l: family = "Matrice series"
        elif "inspire" in name_l: family = "Inspire series"
        elif "mini" in name_l: family = "Mini series"
        elif "avata" in name_l: family = "AVATA FPV series"
        else: family = "UAV platform"
        if "rtk" in name_l:
            extra += ["GNSS/RTK option"]
        else:
            extra += ["consumer/professional UAV"]
        band_hint = "2.4 GHz / 5.8 GHz bands"
    elif any(x in name_l for x in ["vbar", "frsky", "futaba", "taranis", "radiolink", "云卓", "skydroid"]):
        brand = name.split()[0] if " " in name else name
        family = "remote transmitter"
        type_hint = "RF remote-controller telemetry/control link spectrogram"
        band_hint = "2.4 GHz ISM band"
        extra += ["narrowband hopping", "short burst control frames"]
    elif "background" in name_l:
        brand = "environmental"
        family = "background"
        type_hint = "non-drone ambient spectrum"
        band_hint = "wideband noise / incidental signals"
        extra += ["no structured hopping"]

    en = (f"{type_hint} for {brand} {family} [{name}]; typical {band_hint}; "
          f"distinct hopping interval and bandwidth; " + ", ".join(extra))
    zh = (f"{name} 的射频信号时频图；典型工作频段：{band_hint}；具有特定跳频间隔与带宽特征；"
          f"用于无人机/遥控器识别；与同品牌/同系列其他型号在跳频周期、带宽上存在差异")
    return en, zh

EN_TEMPLATES = [
    "A {desc}.",
    "An RF fingerprint: {desc}.",
    "Time-frequency spectrum sample, {desc}.",
    "Label: {name}. {desc}.",
    "A training caption for drone RF recognition: {desc}.",
]

ZH_TEMPLATES = [
    "用于无人机识别的时频图样本：{desc}。",
    "射频指纹：{desc}。",
    "标签：{name}。{desc}。",
    "开放集识别任务的文本描述：{desc}。"
]

def build_prompts(name: str):
    prompts = []
    if USE_AUTO_DESCRIPTORS:
        en_desc, zh_desc = auto_descriptor(name)
    else:
        en_desc, zh_desc = name, name

    if USE_EN_PROMPTS:
        for t in EN_TEMPLATES:
            prompts.append(t.format(name=name, desc=en_desc))
    if USE_ZH_PROMPTS:
        for t in ZH_TEMPLATES:
            prompts.append(t.format(name=name, desc=zh_desc))

    # 极简关键词（帮助形成“语义边界”）
    prompts += [
        f"[CLASS] {name} [MODALITY] RF spectrogram [TASK] open-set recognition",
        f"{name} / RF / control / hopping / bandwidth / interval"
    ]
    return prompts

def encode_with_ensemble(processor, model, prompts):
    """对同一类别的多模板 prompts 编码：逐条L2归一化 -> 求均值 -> 再归一化。"""
    inputs = processor(text=prompts, images=None, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)  # [m, D]
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = torch.nn.functional.normalize(mean_feat, p=2, dim=-1)
    return mean_feat.squeeze(0).cpu()

def pca_whitening(X: np.ndarray, out_dim: Optional[int] = None):
    """PCA 白化（可选降维），并做 L2 归一化。兼容 Python 3.8 的类型注解。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if out_dim is None or out_dim > Vt.shape[0]:
        out_dim = Vt.shape[0]
    V = Vt[:out_dim].T        # [D, out_dim]
    S_diag = S[:out_dim]      # [out_dim]
    eps = 1e-6
    Z = (Xc @ V) / (np.sqrt(S_diag + eps))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) 加载模型
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    torch.set_grad_enabled(False)

    # 2) 先构造并“打印+保存”每类最终使用的 prompts（与你编码时一致）
    all_prompts_by_label = {lbl: build_prompts(lbl) if USE_PROMPT_ENSEMBLE else [lbl]
                            for lbl in drone_categories}

    print("\n================  最终用于编码的 Prompts（逐类）  ================\n")
    for i, lbl in enumerate(drone_categories):
        prompts = all_prompts_by_label[lbl]
        print(f"[{i:02d}] {lbl}  ——  {len(prompts)} prompts")
        for k, p in enumerate(prompts, start=1):
            p_clean = " ".join(p.split())
            print(f"  ({k:02d}) {p_clean}")
        print()

    with open(SAVE_PROMPTS_TXT, "w", encoding="utf-8") as ftxt:
        for i, lbl in enumerate(drone_categories):
            prompts = all_prompts_by_label[lbl]
            ftxt.write(f"[{i:02d}] {lbl}  ——  {len(prompts)} prompts\n")
            for k, p in enumerate(prompts, start=1):
                p_clean = " ".join(p.split())
                ftxt.write(f"  ({k:02d}) {p_clean}\n")
            ftxt.write("\n")
    with open(SAVE_PROMPTS_JSON, "w", encoding="utf-8") as fjson:
        json.dump(all_prompts_by_label, fjson, ensure_ascii=False, indent=2)
    print(f"已保存 prompts 到:\n- {SAVE_PROMPTS_TXT}\n- {SAVE_PROMPTS_JSON}")

    # 3) 编码（与上面保存的一致，保证可复现）
    all_embeds = []
    for name in drone_categories:
        prompts = all_prompts_by_label[name]
        emb = encode_with_ensemble(processor, model, prompts)
        all_embeds.append(emb.numpy())
    E = np.stack(all_embeds, axis=0)   # [N, D]（各类均已L2归一化+均值）

    # 保存原始嵌入
    torch.save(torch.tensor(E), SAVE_EMB_PATH)
    print(f"\n文本向量已保存至: {SAVE_EMB_PATH}")

    # 4) （可选）PCA Whitening 去相关
    if USE_WHITENING:
        E_proc = pca_whitening(E, out_dim=WHITENING_DIM)
        print("已进行 PCA Whitening（含L2归一化）。")
    else:
        E_proc = E / np.linalg.norm(E, axis=1, keepdims=True)

    # 5) 相似度矩阵（余弦）
    S = E_proc @ E_proc.T
    S = np.clip(S, -1.0, 1.0)
    np.save(SAVE_SIM_NPY, S)
    np.savetxt(SAVE_SIM_CSV, S, delimiter=",", fmt="%.6f", encoding="utf-8")
    print(f"相似度矩阵已保存:\n- {SAVE_SIM_NPY}\n- {SAVE_SIM_CSV}")

    # 6) 打印数值统计
    print("\n类别顺序：")
    for i, c in enumerate(drone_categories):
        print(f"{i:2d}. {c}")
    np.set_printoptions(precision=4, suppress=True)
    print("\n相似度矩阵（对角为 1.0 ）：\n", S)

    off_diag = S[~np.eye(S.shape[0], dtype=bool)]
    print("\n非对角元素统计：")
    print(f"min: {off_diag.min():.4f}, max: {off_diag.max():.4f}, "
          f"mean: {off_diag.mean():.4f}, std: {off_diag.std():.4f}")

    # 7) 绘制热力图（带数值）
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r")

    # 用字母 A-X 作为标签
    letters = [chr(ord('A') + i) for i in range(len(drone_categories))]
    ax.set_xticks(range(len(drone_categories)))
    ax.set_yticks(range(len(drone_categories)))
    ax.set_xticklabels(letters, rotation=90, fontsize=10)
    ax.set_yticklabels(letters, fontsize=10)

    # 在格子里标数值
    if ANNOTATE_VALUES and S.shape[0] <= 30:
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=300)
    print(f"热力图已保存: {SAVE_FIG_PATH}")

    # 8) 可选：PCA 2D 可视
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0)
        X2 = pca.fit_transform(E_proc)
        plt.figure(figsize=(8, 7), dpi=150)
        plt.scatter(X2[:,0], X2[:,1], s=50)
        for i, name in enumerate(drone_categories):
            plt.text(X2[i,0]+0.01, X2[i,1]+0.01, name, fontsize=8)
        plt.title("PCA of (whitened) Text Embeddings" if USE_WHITENING else "PCA of Text Embeddings")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(SAVE_PCA_PATH, dpi=300)
        print(f"PCA 散点图已保存: {SAVE_PCA_PATH}")
    except Exception as e:
        print(f"PCA 可视化跳过（可安装 scikit-learn 使用），原因: {e}")

if __name__ == "__main__":
    main()
