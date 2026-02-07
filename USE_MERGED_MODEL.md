# 使用合并后的模型指南

本指南将帮助你将合并后的 LoRA 模型转换为 GGUF 格式并导入到 Ollama。

## 前提条件

- 合并后的模型已保存在 `zsh-model-merged/` 目录
- 已安装 Ollama
- Python 3.8+ 和必要的依赖

## 方法 1: 使用自动化脚本（推荐）

最简单的方法是使用我们提供的自动化脚本：

```bash
./use_merged_model.sh
```

脚本会：
1. 检查合并模型是否存在
2. 提供三种转换方法供选择
3. 自动转换并导入到 Ollama

## 方法 2: 手动转换（Python 脚本）

### 步骤 1: 准备转换脚本

```bash
# 确保有最新的转换脚本
mkdir -p llama.cpp
curl -o llama.cpp/convert_hf_to_gguf.py \
  https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py
chmod +x llama.cpp/convert_hf_to_gguf.py
```

### 步骤 2: 修复兼容性问题

转换脚本可能需要修复以兼容当前的 `gguf` 版本：

```python
# 修复脚本中的导入问题
python3 << 'EOF'
import re
script_path = 'llama.cpp/convert_hf_to_gguf.py'
with open(script_path, 'r') as f:
    content = f.read()

# 修复 MistralTokenizerType 导入
if 'from gguf.vocab import MistralTokenizerType, MistralVocab' in content:
    new_import = '''try:
    from gguf.vocab import MistralTokenizerType, MistralVocab
except ImportError:
    MistralTokenizerType = None
    MistralVocab = None'''
    content = content.replace(
        'from gguf.vocab import MistralTokenizerType, MistralVocab',
        new_import
    )
    with open(script_path, 'w') as f:
        f.write(content)
    print("✅ 已修复")
EOF
```

### 步骤 3: 转换模型

```bash
python3 llama.cpp/convert_hf_to_gguf.py \
  zsh-model-merged \
  --outfile zsh-model-merged/model.gguf
```

### 步骤 4: 导入到 Ollama

```bash
python3 -m model_completer.cli --import-to-ollama
```

## 方法 3: 使用 llama.cpp C++ 工具（最可靠）

### 步骤 1: 编译 llama.cpp

```bash
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 编译转换工具
make convert-hf-to-gguf
```

### 步骤 2: 转换模型

```bash
./convert-hf-to-gguf ../zsh-model-merged --outfile ../zsh-model-merged/model.gguf
```

### 步骤 3: 导入到 Ollama

```bash
cd ..
python3 -m model_completer.cli --import-to-ollama
```

## 方法 4: 使用 Hugging Face（如果已上传）

如果你已经将合并后的模型上传到 Hugging Face 并转换为 GGUF：

```python
python3 -c "
import sys
sys.path.insert(0, 'src')
from model_completer.ollama_lora_import import import_lora_to_ollama

# 替换为你的 HF 仓库 ID 和量化版本
import_lora_to_ollama(hf_gguf_repo='your-username/your-model:Q4_K_M')
"
```

或者直接在 Ollama 中使用：

```bash
ollama run hf.co/your-username/your-model:Q4_K_M
```

## 验证

转换和导入成功后，验证模型：

```bash
# 查看模型信息
ollama show zsh-assistant

# 测试模型
ollama run zsh-assistant "complete: git comm"
```

## 故障排除

### 问题 1: 转换脚本导入错误

**错误**: `ImportError: cannot import name 'MistralTokenizerType'`

**解决**: 使用上面提供的修复脚本，或更新 `gguf` 包：
```bash
pip install --upgrade gguf
```

### 问题 2: 转换失败

**错误**: 转换过程中出现其他错误

**解决**: 
1. 尝试使用 C++ 工具（方法 3）
2. 检查模型文件是否完整
3. 查看详细日志：`/tmp/gguf_convert.log`

### 问题 3: Ollama 导入失败

**错误**: `unsupported architecture`

**解决**: 
- 确保使用的是 GGUF 格式，不是 HF 格式
- 检查 GGUF 文件是否完整
- 尝试重新转换

## 当前状态

如果转换失败，系统会回退到使用 base model (`qwen3:1.7b`) + 优化的 prompts。这些 prompts 已经包含了 LoRA 训练的知识，所以功能仍然可用，只是性能可能不如直接使用合并后的模型。

## 性能对比

- **合并后的模型 (GGUF)**: 最佳性能，使用实际的 LoRA 权重
- **Base model + Prompts**: 良好性能，通过优化的 prompts 编码 LoRA 知识

两种方式都能正常工作，但合并后的模型通常会有更好的准确性和一致性。



