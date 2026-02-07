#!/bin/bash
# Script to convert merged model to GGUF and import to Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGED_MODEL_DIR="$SCRIPT_DIR/zsh-model-merged"
GGUF_FILE="$MERGED_MODEL_DIR/model.gguf"

echo "=== 使用合并后的模型 ==="
echo ""

# Check if merged model exists
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "❌ 合并模型目录不存在: $MERGED_MODEL_DIR"
    echo "💡 请先运行模型合并: python -m model_completer.cli --import-to-ollama"
    exit 1
fi

if [ ! -f "$MERGED_MODEL_DIR/model.safetensors" ]; then
    echo "❌ 合并模型文件不存在"
    exit 1
fi

echo "✅ 合并模型目录存在: $MERGED_MODEL_DIR"
echo ""

# Check if GGUF already exists
if [ -f "$GGUF_FILE" ]; then
    echo "✅ GGUF 文件已存在: $GGUF_FILE"
    echo "   大小: $(du -h "$GGUF_FILE" | cut -f1)"
    echo ""
    echo "🔄 直接导入到 Ollama..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from model_completer.ollama_lora_import import OllamaLoRAImporter

importer = OllamaLoRAImporter()
gguf_file = Path('zsh-model-merged/model.gguf')
if gguf_file.exists():
    modelfile = importer.create_modelfile_from_merged_model(gguf_file)
    if modelfile:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile)
            modelfile_path = f.name
        
        import subprocess
        result = subprocess.run(['ollama', 'create', 'zsh-assistant', '-f', modelfile_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print('✅ 模型已成功导入到 Ollama!')
        else:
            print(f'❌ 导入失败: {result.stderr}')
            exit(1)
"
    exit 0
fi

echo "📋 转换方法选择："
echo ""
echo "方法 1: 使用 llama.cpp Python 脚本（推荐先试这个）"
echo "方法 2: 使用 llama.cpp C++ 工具（更可靠，需要编译）"
echo "方法 3: 上传到 Hugging Face 并使用 hf.co 方式"
echo ""
read -p "请选择方法 (1/2/3): " method

case $method in
    1)
        echo ""
        echo "🔄 方法 1: 使用 Python 转换脚本..."
        echo ""
        
        # Try to fix the convert script
        CONVERT_SCRIPT="$SCRIPT_DIR/llama.cpp/convert_hf_to_gguf.py"
        if [ ! -f "$CONVERT_SCRIPT" ]; then
            echo "📥 下载转换脚本..."
            mkdir -p "$SCRIPT_DIR/llama.cpp"
            curl -o "$CONVERT_SCRIPT" "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
            chmod +x "$CONVERT_SCRIPT"
        fi
        
        # Try to patch the import issue
        echo "🔧 尝试修复转换脚本兼容性..."
        python3 << 'PYTHON_SCRIPT'
import re
script_path = 'llama.cpp/convert_hf_to_gguf.py'
try:
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix MistralTokenizerType import
    if 'from gguf.vocab import MistralTokenizerType, MistralVocab' in content:
        new_import = '''try:
    from gguf.vocab import MistralTokenizerType, MistralVocab
except ImportError:
    # For newer gguf versions, these may not be needed
    MistralTokenizerType = None
    MistralVocab = None'''
        content = content.replace(
            'from gguf.vocab import MistralTokenizerType, MistralVocab',
            new_import
        )
        with open(script_path, 'w') as f:
            f.write(content)
        print("✅ 已修复导入问题")
except Exception as e:
    print(f"⚠️  修复失败: {e}")
PYTHON_SCRIPT
        
        echo ""
        echo "🔄 开始转换（这可能需要几分钟）..."
        python3 "$CONVERT_SCRIPT" "$MERGED_MODEL_DIR" --outfile "$GGUF_FILE" 2>&1 | tee /tmp/gguf_convert.log
        
        if [ -f "$GGUF_FILE" ]; then
            echo ""
            echo "✅ 转换成功! GGUF 文件: $GGUF_FILE"
            echo "   大小: $(du -h "$GGUF_FILE" | cut -f1)"
        else
            echo ""
            echo "❌ 转换失败，请查看日志: /tmp/gguf_convert.log"
            echo "💡 尝试方法 2 或 3"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo "🔄 方法 2: 使用 llama.cpp C++ 工具..."
        echo ""
        
        LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
        
        if [ ! -d "$LLAMA_CPP_DIR" ]; then
            echo "📥 克隆 llama.cpp..."
            git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
        fi
        
        cd "$LLAMA_CPP_DIR"
        
        echo "🔧 编译 llama.cpp..."
        if [ ! -f "convert-hf-to-gguf" ]; then
            make convert-hf-to-gguf || {
                echo "❌ 编译失败"
                echo "💡 请确保已安装: cmake, make, g++"
                exit 1
            }
        fi
        
        echo ""
        echo "🔄 开始转换..."
        ./convert-hf-to-gguf "$MERGED_MODEL_DIR" --outfile "$GGUF_FILE"
        
        if [ -f "$GGUF_FILE" ]; then
            echo ""
            echo "✅ 转换成功! GGUF 文件: $GGUF_FILE"
        else
            echo "❌ 转换失败"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo "📤 方法 3: 上传到 Hugging Face"
        echo ""
        echo "请按照以下步骤："
        echo "1. 上传合并后的模型到 Hugging Face Hub"
        echo "2. 在 HF 上转换为 GGUF 格式（或使用其他工具转换后上传）"
        echo "3. 使用以下命令导入："
        echo ""
        echo "   python3 -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.ollama_lora_import import import_lora_to_ollama
import_lora_to_ollama(hf_gguf_repo='your-username/your-model:Q4_K_M')
\""
        echo ""
        read -p "按 Enter 继续..."
        exit 0
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

# Import to Ollama
echo ""
echo "🔄 导入到 Ollama..."
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from model_completer.ollama_lora_import import OllamaLoRAImporter
import subprocess
import tempfile

importer = OllamaLoRAImporter()
gguf_file = Path('zsh-model-merged/model.gguf')

if not gguf_file.exists():
    print('❌ GGUF 文件不存在')
    sys.exit(1)

print(f'✅ 找到 GGUF 文件: {gguf_file}')
modelfile = importer.create_modelfile_from_merged_model(gguf_file)

if modelfile:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
        f.write(modelfile)
        modelfile_path = f.name
    
    print('🔄 创建 Ollama 模型...')
    result = subprocess.run(
        ['ollama', 'create', 'zsh-assistant', '-f', modelfile_path],
        capture_output=True,
        text=True
    )
    
    import os
    os.unlink(modelfile_path)
    
    if result.returncode == 0:
        print('✅ 模型已成功导入到 Ollama!')
        print('')
        print('现在可以使用合并后的模型了:')
        print('  ollama run zsh-assistant')
    else:
        print(f'❌ 导入失败: {result.stderr}')
        if result.stdout:
            print(f'stdout: {result.stdout}')
        sys.exit(1)
else:
    print('❌ 无法创建 Modelfile')
    sys.exit(1)
PYTHON_SCRIPT

echo ""
echo "✅ 完成! 现在可以使用合并后的模型了"
echo ""
echo "验证:"
echo "  ollama show zsh-assistant"
echo ""
echo "测试:"
echo "  ollama run zsh-assistant 'complete: git comm'"



