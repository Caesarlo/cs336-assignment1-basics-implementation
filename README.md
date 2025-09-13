# CS336 Spring 2025 Assignment 1: Transformer 基础实现

> **作业目标**: 从零开始实现 Transformer 模型的核心组件，包括注意力机制、前馈网络、层归一化等。

## 📋 作业概览

这个作业要求我们手动实现 Transformer 模型的基础组件，而不是直接使用 PyTorch 的高级 API。通过这个作业，你将深入理解：

- **注意力机制** (Scaled Dot-Product Attention)
- **多头注意力** (Multi-Head Attention) 
- **前馈网络** (Feed-Forward Network)
- **层归一化** (Layer Normalization)
- **位置编码** (Positional Encoding)
- **完整的 Transformer 块**

## 🚀 快速开始

### 1. 环境配置

本项目使用 `uv` 来管理 Python 环境和依赖，确保环境的一致性和可重现性。

**安装 uv**:
```bash
# 推荐方式：从官网下载
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用包管理器
pip install uv
# 或者 brew install uv (macOS)
```

**运行代码**:
```bash
# 使用 uv 运行任何 Python 文件，环境会自动配置
uv run <python_file_path>
```

### 2. 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_specific.py

# 运行测试并显示详细输出
uv run pytest -v
```

**⚠️ 重要提示**: 初始运行时，所有测试都会失败并抛出 `NotImplementedError`，这是正常的！你需要完成 `tests/adapters.py` 中的函数来连接你的实现。

### 3. 下载数据集

```bash
# 创建数据目录
mkdir -p data
cd data

# 下载 TinyStories 数据集（用于训练）
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 下载 OpenWebText 样本数据集
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 📁 项目结构

```
cs336_basics/
├── cs336_basics/
│   ├── transformer/
│   │   └── module.py          # 🔥 主要实现文件 - 在这里写你的代码！
│   └── pretokenization_example.py
├── tests/
│   └── adapters.py            # 🔗 测试适配器 - 连接你的实现和测试
├── data/                      # 📊 数据集存放目录
├── pyproject.toml            # 📦 项目配置和依赖
└── README.md                 # 📖 这个文件
```

## 🎯 实现指南

### 核心文件说明

1. **`cs336_basics/transformer/module.py`** - 你的主要工作文件
   - 包含所有需要实现的 Transformer 组件
   - 每个类都有清晰的接口定义
   - 需要你从零开始实现前向传播逻辑

2. **`tests/adapters.py`** - 测试适配器
   - 连接你的实现和测试框架
   - 包含各种测试辅助函数
   - 需要你完成这些函数来让测试通过

### 实现顺序建议

1. **基础组件** (从简单到复杂):
   - `Linear` - 线性层
   - `Embedding` - 词嵌入层
   - `LayerNorm` - 层归一化

2. **注意力机制**:
   - `ScaledDotProductAttention` - 缩放点积注意力
   - `MultiHeadAttention` - 多头注意力

3. **前馈网络**:
   - `Swiglu` - SwiGLU 激活函数

4. **完整模块**:
   - `TransformerBlock` - Transformer 块
   - `Transformer` - 完整模型

### 调试技巧

```bash
# 运行单个测试来调试特定组件
uv run pytest tests/test_specific.py::test_function_name -v

# 使用 Python 调试器
uv run python -m pdb your_script.py

# 检查张量形状和数值
# 在代码中添加 print() 语句来调试
```

## 📚 学习资源

- **作业详细说明**: [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)
- **PyTorch 文档**: https://pytorch.org/docs/
- **Attention 论文**: "Attention Is All You Need" (Vaswani et al., 2017)
- **uv 使用指南**: https://docs.astral.sh/uv/guides/projects/

## 🐛 问题反馈

如果在作业过程中遇到问题，可以：
1. 检查 GitHub Issues 看是否有类似问题
2. 创建新的 Issue 描述你的问题
3. 提交 Pull Request 修复发现的 bug

## 💡 提示

- 仔细阅读每个函数的文档字符串，它们包含了重要的实现细节
- 使用 `einops` 库来简化张量操作
- 注意张量的维度顺序，PyTorch 通常使用 `(batch_size, seq_len, d_model)`
- 测试你的实现时，确保输出张量的形状正确
- 使用 `torch.nn.functional` 中的函数来实现激活函数等操作


