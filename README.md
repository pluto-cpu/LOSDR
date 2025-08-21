# LOSDR


# CLIP-based Drone Text Embedding & Similarity Visualization  ---text_extract.py

本项目使用 [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) 模型，对无人机/遥控器类别构造文本 Prompt，提取文本嵌入，并计算类别间的语义相似度矩阵。  
在此基础上，生成 **热力图** 和 **PCA 可视化散点图**，用于分析类别之间的语义分布。

## ✨ 功能特点

- **多模板 Prompt 集成 (Prompt Ensemble)**  
  为每个类别生成多条中英文 prompt，并进行平均编码，增强语义表示的稳定性。

- **自动差异化描述 (Auto Descriptor)**  
  自动在类别名之外加入品牌、系列、频段、信号形态等差异化信息，降低不同类别间过高的相似度。

- **中英文双语融合**  
  提供中英文 prompt，增强跨语言鲁棒性。

- **关键词锚点 (Semantic Anchors)**  
  加入任务相关的简短关键词，如 RF / spectrogram / hopping / bandwidth 等，突出模态信息和任务背景。

- **PCA Whitening (可选)**  
  去除公共成分，使类别嵌入更加分散，提升区分度。

- **可视化输出**  
  - 相似度矩阵热力图（支持数值标注、字母 A–X 作为类别索引）
  - PCA 降维散点图
  - 类别对应的最终 Prompt 列表（TXT/JSON 保存）

---

## 📂 项目结构

