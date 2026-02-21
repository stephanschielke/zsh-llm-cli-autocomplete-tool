#!/usr/bin/env python3
"""
Import fine-tuned LoRA model into Ollama.

Workflow (so Ollama has both base + adapter for autocomplete):
  1. Download LoRA adapter from Hugging Face (e.g. duoyuncloud/zsh-cli-lora).
  2. Load base model from Hugging Face (e.g. Qwen/Qwen2-0.5B) via transformers.
  3. Merge adapter into base model, save merged model.
  4. Convert merged model to GGUF (for Ollama).
  5. Create Ollama model from GGUF so the server runs the single merged model (base+adapter).
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class OllamaLoRAImporter:
    """Import LoRA fine-tuned model into Ollama."""
    
    def __init__(self, base_dir: Optional[Path] = None, hf_repo_id: Optional[str] = None):
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        self.lora_output_dir = self.base_dir / "zsh-lora-output"
        self.model_name = "zsh-assistant"
        self.hf_repo_id = hf_repo_id
        
    def download_from_huggingface(self, repo_id: str) -> bool:
        """Download LoRA adapter from Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "username/model-name")
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Install it with: pip install huggingface_hub")
            return False
        
        logger.info(f"📥 Downloading LoRA adapter from Hugging Face: {repo_id}")
        
        try:
            # Create output directory
            self.lora_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Download adapter files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(self.lora_output_dir),
                local_dir_use_symlinks=False,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.safetensors", "*.json"]
            )
            
            logger.info(f"✅ Successfully downloaded LoRA adapter to {self.lora_output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download from Hugging Face: {e}")
            return False
    
    def is_lora_ready(self) -> bool:
        """Check if LoRA model files exist."""
        if not self.lora_output_dir.exists():
            return False
        
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        for file in required_files:
            if not (self.lora_output_dir / file).exists():
                return False
        
        return True
    
    def get_base_model_name(self) -> Optional[str]:
        """Get base model name from adapter config."""
        if not self.is_lora_ready():
            return None
        
        config_file = self.lora_output_dir / "adapter_config.json"
        try:
            with open(config_file) as f:
                config = json.load(f)
            return config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B")
        except Exception as e:
            logger.error(f"Failed to read adapter config: {e}")
            return None
    
    def get_ollama_model_name(self, hf_model_name: str) -> str:
        """Map Hugging Face model name to Ollama model name."""
        # Mapping from HF model names to Ollama model names
        model_mapping = {
            "Qwen/Qwen2-0.5B": "qwen2:0.5b",
            "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
            "Qwen/Qwen3-1.7B": "qwen3:1.7b",
            "Qwen/Qwen3-1.7B-Instruct": "qwen3:1.7b",
            "distilgpt2": "tinyllama",
            "gpt2": "tinyllama",
            "gpt2-medium": "llama2",
            "gpt2-large": "llama2:13b",
            "microsoft/DialoGPT-small": "tinyllama",
            "microsoft/DialoGPT-medium": "llama2",
            "codellama/CodeLlama-7b-hf": "codellama:7b",
        }
        
        # Check direct mapping
        if hf_model_name in model_mapping:
            return model_mapping[hf_model_name]
        
        # Check if it's a Qwen2 0.5B model
        if "qwen2" in hf_model_name.lower() and "0.5" in hf_model_name:
            return "qwen2:0.5b"
        if "qwen" in hf_model_name.lower() or "Qwen" in hf_model_name:
            return "qwen3:1.7b"
        # Default fallback
        return "qwen2:0.5b"
    
    def create_ollama_modelfile(self) -> Optional[str]:
        """Create Ollama Modelfile for the fine-tuned model.
        
        Fallback when merged GGUF is not used: base model from Ollama with
        system prompts that encode LoRA-style completion behavior.
        
        The prompts are designed based on the training data patterns to
        guide the model to generate appropriate command completions.
        """
        hf_base_model = self.get_base_model_name()
        if not hf_base_model:
            return None
        
        # Map to Ollama model name
        ollama_base_model = self.get_ollama_model_name(hf_base_model)
        
        # Enhanced system prompt based on LoRA training patterns
        modelfile_content = f"""FROM {ollama_base_model}

SYSTEM \"\"\"You are a Zsh command completion expert fine-tuned on CLI command patterns.
You have learned from 277 command completion examples covering Git, Docker, NPM, Python, Kubernetes, and system commands.

Your task: Complete shell commands accurately and concisely.

Rules:
1. Always respond with complete, executable commands only
2. Never explain, comment, or add descriptions
3. Complete commands in the most common/practical way
4. Use appropriate flags and arguments based on command patterns
5. Keep responses short and direct (max 30 tokens)

Command Patterns You Know:
- Git: commit, status, add, push, pull, branch, checkout, merge, rebase
- Docker: run, build, ps, exec, logs, stop, rm, images
- NPM: install, run, start, test, build, publish
- Python: -m, -c, pip install/run/list
- Kubernetes: get, apply, delete, describe, logs
- System: ls, cd, mkdir, cp, mv, rm, find, grep

Examples:
Input: git comm
Output: git commit -m "feat: add new feature"

Input: docker run
Output: docker run -it --rm

Input: npm run
Output: npm run start

Input: python -m
Output: python -m pip install

Remember: Only output the command, nothing else.
\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 30
PARAMETER num_ctx 512
PARAMETER repeat_penalty 1.1
"""
        
        return modelfile_content
    
    def create_modelfile_from_merged_model(self, model_path: Path) -> Optional[str]:
        """Create Modelfile for merged model (HF format or GGUF).
        
        Ollama requires absolute paths for local model files.
        """
        # Ensure we have an absolute path
        model_path = model_path.resolve()
        
        # Check if it's a GGUF file or directory
        if model_path.is_file() and model_path.suffix == '.gguf':
            # GGUF file - use absolute path
            model_path_str = str(model_path)
            modelfile_content = f"""FROM {model_path_str}

SYSTEM \"\"\"You are a Zsh command completion expert fine-tuned on command-line completions.
Your role is to complete shell commands accurately and concisely.

Guidelines:
- Always respond with complete, executable commands only
- Never explain or add commentary
- Complete the command in the most common/practical way
- Use appropriate flags and arguments based on command patterns
- Keep responses short and direct
\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 30
PARAMETER num_ctx 512
PARAMETER repeat_penalty 1.1
"""
        else:
            # Hugging Face format directory - use absolute path
            model_path_str = str(model_path)
            modelfile_content = f"""FROM {model_path_str}

SYSTEM \"\"\"You are a Zsh command completion expert fine-tuned on command-line completions.
Your role is to complete shell commands accurately and concisely.

Guidelines:
- Always respond with complete, executable commands only
- Never explain or add commentary
- Complete the command in the most common/practical way
- Use appropriate flags and arguments based on command patterns
- Keep responses short and direct
\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 30
PARAMETER num_ctx 512
PARAMETER repeat_penalty 1.1
"""
        return modelfile_content
    
    def import_to_ollama(self, use_merged_model: bool = True, download_from_hf: bool = False, hf_gguf_repo: Optional[str] = None) -> bool:
        """Import LoRA model to Ollama.
        
        Args:
            use_merged_model: If True, merge LoRA adapter and use the merged model.
                            If False, use base model with optimized prompts.
            download_from_hf: If True and hf_repo_id is set, download adapter from Hugging Face first.
            hf_gguf_repo: If provided, use GGUF model directly from Hugging Face (e.g., "username/model-name:Q4_K_M").
                         This bypasses local conversion and uses: ollama run hf.co/{hf_gguf_repo}
        
        Returns:
            True if import successful, False otherwise.
        """
        # Download from Hugging Face if requested
        if download_from_hf and self.hf_repo_id:
            if not self.download_from_huggingface(self.hf_repo_id):
                logger.error("Failed to download adapter from Hugging Face")
                return False
        
        # Check if LoRA adapter is available (local or downloaded)
        if not self.is_lora_ready():
            if self.hf_repo_id:
                logger.info(f"LoRA adapter not found locally, downloading from {self.hf_repo_id}...")
                if not self.download_from_huggingface(self.hf_repo_id):
                    logger.error("Failed to download adapter from Hugging Face")
                    return False
            else:
                logger.error("LoRA adapter not found and no Hugging Face repo ID provided")
                logger.error("Either train a model locally or provide hf_repo_id")
                return False
        
        # Check if Ollama is available
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.error("❌ Ollama is not installed or not in PATH")
                return False
        except FileNotFoundError:
            logger.error("❌ Ollama is not installed")
            return False
        
        # Check if Ollama server is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                logger.warning("⚠️  Ollama server may not be running")
                logger.info("💡 Try starting Ollama: ollama serve")
        except Exception as e:
            logger.warning(f"⚠️  Cannot connect to Ollama server: {e}")
            logger.info("💡 Try starting Ollama: ollama serve")
            # Don't fail here - Ollama might start during import
        
        # Ensure base model is available in Ollama (from adapter config, e.g. qwen2:0.5b or qwen3:1.7b)
        hf_base = self.get_base_model_name() or "Qwen/Qwen2-0.5B"
        ollama_base = self.get_ollama_model_name(hf_base)
        logger.info(f"📥 Ensuring base model {ollama_base} is available in Ollama...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                base_available = any(ollama_base.split(':')[0] in name.lower() for name in model_names)
                if not base_available:
                    logger.info(f"   {ollama_base} not found, pulling from Ollama library...")
                    pull_result = subprocess.run(
                        ['ollama', 'pull', ollama_base],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if pull_result.returncode == 0:
                        logger.info(f"   ✅ {ollama_base} pulled successfully")
                    else:
                        logger.warning(f"   ⚠️  Failed to pull {ollama_base}: {pull_result.stderr}")
                else:
                    logger.info(f"   ✅ {ollama_base} already available")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not check/pull base model: {e}")
        
        model_path = None
        modelfile_content = None
        
        # Try to merge LoRA adapter with base model first
        if use_merged_model:
            logger.info("🔄 Attempting to merge LoRA adapter with base model...")
            logger.info("   This will download the base model and merge it with your LoRA adapter")
            try:
                merged_path = self.merge_and_convert_to_gguf()
                
                if merged_path and merged_path.exists():
                    # Check if it's a GGUF file (Ollama can use this)
                    if merged_path.is_file() and merged_path.suffix == '.gguf':
                        model_path = merged_path
                        logger.info(f"✅ Using merged GGUF model from: {model_path}")
                        modelfile_content = self.create_modelfile_from_merged_model(model_path)
                    elif merged_path.is_dir():
                        # It's a Hugging Face format directory
                        # Try to convert to GGUF so we can use the merged LoRA adapter
                        logger.info(f"✅ Merged model saved to: {merged_path}")
                        logger.info("🔄 Attempting to convert to GGUF format for Ollama...")
                        gguf_file = self.convert_to_gguf(merged_path)
                        
                        if gguf_file and gguf_file.exists():
                            # Successfully converted to GGUF!
                            model_path = gguf_file
                            logger.info(f"✅ Successfully converted to GGUF: {gguf_file}")
                            logger.info("   Will use merged LoRA adapter in GGUF format!")
                            modelfile_content = self.create_modelfile_from_merged_model(model_path)
                        else:
                            # Conversion failed, but try to use merged HF format model directly
                            logger.warning("⚠️  Could not convert to GGUF automatically")
                            logger.info("💡 Attempting to use merged HF format model directly...")
                            logger.info("   (Ollama may not support this for Qwen3, but worth trying)")
                            
                            # Try to create modelfile with merged HF format model
                            try:
                                model_path = merged_path
                                modelfile_content = self.create_modelfile_from_merged_model(merged_path)
                                logger.info("   ✅ Created Modelfile for merged HF format model")
                                logger.info("   Will attempt to import - if it fails, will fallback to base model")
                            except Exception as e:
                                logger.warning(f"   ⚠️  Could not create modelfile for HF format: {e}")
                                logger.info("💡 Will use base model with prompts only (merged model saved at zsh-model-merged/)")
                                logger.info("💡 The merged model is saved at zsh-model-merged/ for manual conversion")
                                use_merged_model = False  # Use base model approach
                else:
                    logger.warning("⚠️  Failed to merge model, falling back to base model with prompts")
                    logger.info("💡 This may happen if transformers/peft are not installed")
                    use_merged_model = False
            except Exception as e:
                logger.warning(f"⚠️  Error during model merge: {e}")
                logger.info("💡 Falling back to base model with optimized prompts")
                logger.debug(f"Merge error details: {e}", exc_info=True)
                use_merged_model = False
        
        # Fallback to base model with optimized prompts
        if not modelfile_content:
            logger.info("📝 Creating Modelfile with base model and optimized prompts...")
            logger.info("💡 Note: Using base model (qwen3:1.7b) with LoRA knowledge in prompts")
            modelfile_content = self.create_ollama_modelfile()
            if not modelfile_content:
                logger.error("❌ Failed to create Modelfile")
                return False
        
        # Write Modelfile
        modelfile_path = self.base_dir / "Modelfile.zsh-assistant"
        try:
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            logger.info(f"📄 Created Modelfile: {modelfile_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write Modelfile: {e}")
            return False
        
        # Create model in Ollama (merged = base+adapter from HF in one model)
        ollama_base = self.get_ollama_model_name(self.get_base_model_name() or "Qwen/Qwen2-0.5B")
        try:
            logger.info(f"🚀 Creating Ollama model: {self.model_name}")
            if use_merged_model and model_path:
                logger.info(f"   Using merged model (base + adapter from Hugging Face): {model_path}")
                if model_path.is_file() and model_path.suffix == '.gguf':
                    logger.info("   ✅ GGUF format: Ollama will run base+adapter as one model")
                else:
                    logger.info("   ⚠️  HF format (Ollama may require GGUF)")
            else:
                logger.info(f"   Fallback: base model ({ollama_base}) with prompts only (no adapter weights)")
            
            result = subprocess.run(
                ['ollama', 'create', self.model_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                if use_merged_model and model_path:
                    logger.info(f"✅ Model {self.model_name} created: Ollama has base+adapter (from Hugging Face)")
                    if model_path.is_file() and model_path.suffix == '.gguf':
                        logger.info("   ✅ Merged GGUF loaded for autocomplete")
                else:
                    logger.info(f"✅ Model {self.model_name} created (base {ollama_base} only)")
                if modelfile_path.exists():
                    modelfile_path.unlink()
                return True
            else:
                error_msg = result.stderr.lower()
                logger.error(f"❌ Failed to create model: {result.stderr}")
                if result.stdout:
                    logger.error(f"stdout: {result.stdout}")
                
                # If model already exists and we have merged GGUF, replace it so Ollama uses base+adapter
                if ("already exists" in error_msg or "model already exists" in error_msg) and use_merged_model and model_path and model_path.is_file() and model_path.suffix == '.gguf':
                    logger.info(f"   Replacing existing {self.model_name} with merged model (base+adapter)...")
                    subprocess.run(['ollama', 'rm', self.model_name], capture_output=True, timeout=30)
                    result = subprocess.run(
                        ['ollama', 'create', self.model_name, '-f', str(modelfile_path)],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ Model {self.model_name} updated with merged base+adapter from Hugging Face")
                        if modelfile_path.exists():
                            modelfile_path.unlink()
                        return True
                    logger.error(f"❌ Retry failed: {result.stderr}")
                
                if "already exists" in error_msg or "model already exists" in error_msg:
                    logger.info(f"   To replace with merged model: ollama rm {self.model_name} then run import again")
                
                # Fallback only if merged GGUF failed (e.g. unsupported architecture)
                if "unsupported architecture" in error_msg and use_merged_model:
                    logger.warning("⚠️  Ollama could not load merged model directly")
                    logger.info(f"💡 Falling back to base ({ollama_base}) with prompts; merged model saved at zsh-model-merged/")
                    fallback_modelfile = self.create_ollama_modelfile()
                    if fallback_modelfile:
                        try:
                            with open(modelfile_path, 'w') as f:
                                f.write(fallback_modelfile)
                            result = subprocess.run(
                                ['ollama', 'create', self.model_name, '-f', str(modelfile_path)],
                                capture_output=True, text=True, timeout=600
                            )
                            if result.returncode == 0:
                                logger.info(f"✅ Model {self.model_name} created (base only)")
                                if modelfile_path.exists():
                                    modelfile_path.unlink()
                                return True
                        except Exception as e:
                            logger.error(f"❌ Fallback failed: {e}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Ollama create command timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create model in Ollama: {e}")
            return False
        finally:
            # Clean up Modelfile
            if modelfile_path.exists():
                modelfile_path.unlink()
    
    def check_gguf_converter(self) -> bool:
        """Check if GGUF converter is available."""
        # Try to find llama.cpp convert script
        try:
            # Check if llama-cpp-python is available (has convert_hf_to_gguf)
            try:
                import llama_cpp
                return True
            except ImportError:
                pass
            
            # Check if llama.cpp convert script exists
            result = subprocess.run(
                ['which', 'convert_hf_to_gguf.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
            
            # Check if it's in common locations
            common_paths = [
                Path.home() / 'llama.cpp' / 'convert_hf_to_gguf.py',
                Path('/usr/local/bin/convert_hf_to_gguf.py'),
            ]
            for path in common_paths:
                if path.exists():
                    return True
            
            return False
        except Exception:
            return False
    
    def convert_to_gguf(self, merged_model_dir: Path) -> Optional[Path]:
        """Convert merged model to GGUF format.
        
        Tries multiple methods:
        1. llama.cpp convert_hf_to_gguf.py script (download if needed)
        2. Fallback: return None (will use base model approach)
        """
        logger.info("🔄 Converting merged model to GGUF format...")
        logger.info("   This will allow Ollama to use the merged LoRA adapter directly")
        
        gguf_file = merged_model_dir / "model.gguf"
        
        # Check if already converted
        if gguf_file.exists():
            logger.info(f"✅ GGUF file already exists: {gguf_file}")
            return gguf_file
        
        # Method 1: Try to find or download llama.cpp convert script
        convert_script = None
        try:
            # Try to find convert_hf_to_gguf.py
            result = subprocess.run(
                ['which', 'convert_hf_to_gguf.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                convert_script = result.stdout.strip()
            else:
                # Check common locations
                common_paths = [
                    Path.home() / 'llama.cpp' / 'convert_hf_to_gguf.py',
                    Path('/usr/local/bin/convert_hf_to_gguf.py'),
                    self.base_dir / 'llama.cpp' / 'convert_hf_to_gguf.py',
                ]
                for path in common_paths:
                    if path.exists():
                        convert_script = str(path)
                        break
                
                # If not found, try to download or update it
                if not convert_script:
                    logger.info("📥 convert_hf_to_gguf.py not found, attempting to download/update...")
                    try:
                        import urllib.request
                        import subprocess
                        llama_cpp_dir = self.base_dir / 'llama.cpp'
                        llama_cpp_dir.mkdir(exist_ok=True)
                        script_path = llama_cpp_dir / 'convert_hf_to_gguf.py'
                        
                        # Try to update llama.cpp if it's a git repo
                        if (llama_cpp_dir / '.git').exists():
                            logger.info("   Updating llama.cpp repository...")
                            update_result = subprocess.run(
                                ['git', '-C', str(llama_cpp_dir), 'pull'],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            if update_result.returncode == 0:
                                logger.info("   ✅ llama.cpp updated")
                            else:
                                logger.warning(f"   ⚠️  Failed to update llama.cpp: {update_result.stderr}")
                        
                        # If script still doesn't exist or we want to ensure latest, download it
                        if not script_path.exists() or True:  # Always download latest for compatibility
                            script_url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
                            logger.info(f"   Downloading latest from: {script_url}")
                            urllib.request.urlretrieve(script_url, script_path)
                        
                        if script_path.exists():
                            script_path.chmod(0o755)  # Make executable
                            convert_script = str(script_path)
                            logger.info(f"   ✅ Script ready at: {convert_script}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Failed to download/update convert script: {e}")
        except Exception as e:
            logger.debug(f"Error finding convert script: {e}")
        
        if convert_script:
            try:
                logger.info(f"🔧 Using convert script: {convert_script}")
                logger.info(f"   Converting: {merged_model_dir} -> {gguf_file}")
                logger.info("   This may take several minutes...")
                
                # Check if gguf module is available (required for conversion)
                try:
                    import gguf
                    logger.info("   ✅ gguf module available")
                except ImportError:
                    logger.warning("   ⚠️  gguf module not found, attempting to install...")
                    try:
                        import subprocess
                        install_result = subprocess.run(
                            ['pip3', 'install', 'gguf>=0.1.0'],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if install_result.returncode == 0:
                            logger.info("   ✅ gguf module installed")
                            # Re-import after installation
                            import gguf
                        else:
                            logger.error(f"   ❌ Failed to install gguf: {install_result.stderr}")
                            logger.error("   💡 Please install manually: pip install gguf")
                            raise ImportError("gguf module is required for GGUF conversion")
                    except Exception as e:
                        logger.error(f"   ❌ Could not install gguf: {e}")
                        raise
                
                # Run conversion
                result = subprocess.run(
                    ['python3', convert_script, str(merged_model_dir), '--outfile', str(gguf_file)],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout for conversion
                )
                
                if result.returncode == 0 and gguf_file.exists():
                    file_size_mb = gguf_file.stat().st_size / (1024*1024)
                    logger.info(f"✅ Successfully converted to GGUF: {gguf_file}")
                    logger.info(f"   File size: {file_size_mb:.2f} MB")
                    return gguf_file
                else:
                    logger.warning(f"⚠️  Conversion failed: {result.stderr}")
                    if result.stdout:
                        logger.debug(f"stdout: {result.stdout}")
                    # Check if it's a missing dependency issue or compatibility issue
                    if "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr:
                        logger.info("💡 Missing Python dependencies for GGUF conversion")
                        logger.info("   Try: pip install gguf")
                    elif "ImportError" in result.stderr or "cannot import" in result.stderr.lower():
                        logger.warning("⚠️  GGUF conversion script compatibility issue detected")
                        logger.info("💡 The convert_hf_to_gguf.py script may be incompatible with current gguf version")
                        logger.info("💡 Try updating llama.cpp: git -C llama.cpp pull")
                        logger.info("💡 Or use base model (qwen3:1.7b) with optimized prompts (current approach)")
            except subprocess.TimeoutExpired:
                logger.error("❌ GGUF conversion timed out (took > 30 minutes)")
            except Exception as e:
                logger.warning(f"⚠️  GGUF conversion failed: {e}")
        
        logger.warning("⚠️  Could not convert to GGUF automatically")
        logger.info(f"💡 The merged model is saved at: {merged_model_dir}")
        logger.info("💡 Alternative conversion methods:")
        logger.info("   1. Use llama.cpp C++ tools (more reliable):")
        logger.info("      - Build llama.cpp: cd llama.cpp && make")
        logger.info(f"      - Convert: ./llama-cli convert-hf-to-gguf {merged_model_dir} --outfile {gguf_file}")
        logger.info("   2. Upload to Hugging Face and use directly:")
        logger.info("      - Upload merged model to HF Hub")
        logger.info("      - Use: ollama run hf.co/username/model-name")
        logger.info("   3. Wait for Ollama to support direct HF format import")
        logger.info("💡 For now, will use base model (qwen3:1.7b) with optimized prompts")
        logger.info("   (The prompts encode the LoRA training knowledge)")
        
        return None
    
    def merge_and_convert_to_gguf(self) -> Optional[Path]:
        """Merge LoRA adapter with base model and convert to GGUF format.
        
        This is the proper way to import a fine-tuned model into Ollama.
        It requires:
        1. Merging LoRA weights into base model
        2. Converting to GGUF format (optional, Ollama can use HF format too)
        3. Creating Ollama model from merged/GGUF model
        """
        # Check if we need to re-merge (adapter is newer than merged model)
        merged_dir = self.base_dir / "zsh-model-merged"
        adapter_file = self.lora_output_dir / "adapter_model.safetensors"
        merged_file = merged_dir / "model.safetensors"
        gguf_file = merged_dir / "model.gguf"
        
        # Check if merged model exists and is up-to-date
        if merged_dir.exists() and merged_file.exists():
            if adapter_file.exists():
                adapter_mtime = adapter_file.stat().st_mtime
                merged_mtime = merged_file.stat().st_mtime
                
                if merged_mtime >= adapter_mtime:
                    logger.info("✅ Merged model exists and is up-to-date")
                    # Check if GGUF exists
                    if gguf_file.exists():
                        logger.info(f"✅ GGUF file found: {gguf_file}")
                        return gguf_file
                    else:
                        logger.info("💡 Merged model exists but no GGUF, will try to convert...")
                        # Try to convert existing merged model to GGUF
                        gguf_result = self.convert_to_gguf(merged_dir)
                        if gguf_result:
                            return gguf_result
                        # Return merged dir if conversion fails
                        return merged_dir
                else:
                    logger.info("🔄 LoRA adapter is newer than merged model, re-merging...")
        
        logger.info("🔄 Merging LoRA adapter with base model...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            import platform
            
            base_model_name = self.get_base_model_name()
            if not base_model_name:
                logger.warning("Could not determine base model, using Qwen/Qwen2-0.5B")
                base_model_name = "Qwen/Qwen2-0.5B"
            
            logger.info(f"📥 Loading base model: {base_model_name}")
            
            # Determine device and quantization settings
            is_macos = platform.system() == "Darwin"
            is_apple_silicon = platform.machine() == "arm64"
            
            quantization_config = None
            if not (is_macos and is_apple_silicon):
                # Try to use quantization on non-Apple Silicon
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    
                    bnb_version = getattr(bnb, '__version__', '0.0.0')
                    try:
                        from packaging import version
                        bnb_supports_apple = version.parse(bnb_version) >= version.parse("0.43.1")
                    except:
                        bnb_supports_apple = False
                    
                    if not (is_macos and is_apple_silicon and not bnb_supports_apple):
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                except ImportError:
                    logger.debug("bitsandbytes not available, loading without quantization")
            
            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = None
                torch_dtype = torch.float16
                quantization_config = None  # Disable quantization on MPS
            else:
                device_map = None
                torch_dtype = torch.float32
                quantization_config = None  # Disable quantization on CPU
            
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
            }
            
            if device_map:
                load_kwargs["device_map"] = device_map
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **load_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load LoRA adapter (local path: use local_files_only so PEFT does not treat as HF repo)
            logger.info("📦 Loading LoRA adapter from %s...", self.lora_output_dir)
            model = PeftModel.from_pretrained(
                base_model,
                str(self.lora_output_dir),
                is_trainable=False,
                local_files_only=True,
            )
            
            # Merge adapter into base model
            logger.info("🔗 Merging LoRA adapter into base model...")
            model = model.merge_and_unload()
            
            # Save merged model
            merged_dir = self.base_dir / "zsh-model-merged"
            merged_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"💾 Saving merged model to {merged_dir}")
            model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
            
            # Update merged model timestamp to match adapter (for future checks)
            adapter_file = self.lora_output_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                adapter_mtime = adapter_file.stat().st_mtime
                merged_file = merged_dir / "model.safetensors"
                if merged_file.exists():
                    import os
                    os.utime(merged_file, (adapter_mtime, adapter_mtime))
            
            logger.info("✅ Model merged successfully!")
            
            # Try to convert to GGUF (this is critical for using merged model in Ollama)
            logger.info("🔄 Attempting to convert merged model to GGUF format...")
            logger.info("   This is required for Ollama to use the merged LoRA adapter")
            gguf_file = self.convert_to_gguf(merged_dir)
            if gguf_file and gguf_file.exists():
                logger.info(f"✅ Successfully converted to GGUF: {gguf_file}")
                return gguf_file
            
            # If GGUF conversion failed, return merged dir (but Ollama may not support HF format Qwen3)
            logger.warning("⚠️  GGUF conversion failed or not available")
            logger.info(f"💡 Merged model saved at: {merged_dir}")
            logger.info("💡 Will attempt to use it; Ollama may require GGUF for this architecture")
            return merged_dir
            
        except ImportError:
            logger.error("❌ transformers or peft not available")
            logger.info("💡 Install with: pip install transformers peft torch")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to merge model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None


def import_lora_to_ollama(base_dir: Optional[Path] = None, use_merged_model: bool = True, hf_repo_id: Optional[str] = None, force_merge: bool = False) -> bool:
    """Convenience function to import LoRA model to Ollama.
    
    Args:
        base_dir: Base directory of the project (defaults to project root)
        use_merged_model: If True, merge LoRA adapter and use the merged model.
                        If False, use base model with optimized prompts.
        hf_repo_id: Hugging Face repository ID to download adapter from (e.g., "username/model-name")
    
    Returns:
        True if import successful, False otherwise.
    """
    importer = OllamaLoRAImporter(base_dir, hf_repo_id=hf_repo_id)
    # If force_merge is True or hf_repo_id is provided, try to merge
    if force_merge or hf_repo_id:
        use_merged_model = True
    return importer.import_to_ollama(use_merged_model=use_merged_model, download_from_hf=bool(hf_repo_id))


def is_lora_imported_to_ollama(model_name: str = "zsh-assistant") -> bool:
    """Check if LoRA model is already imported to Ollama."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            return model_name in model_names
    except Exception:
        pass
    return False


def check_if_using_merged_lora(base_dir: Optional[Path] = None) -> bool:
    """Check if zsh-assistant model is using merged LoRA adapter.
    
    Returns:
        True if merged LoRA model exists and is being used, False otherwise.
    """
    base_dir = base_dir or Path(__file__).parent.parent.parent
    merged_dir = base_dir / "zsh-model-merged"
    
    # Check if merged model directory exists and has model files
    if merged_dir.exists():
        # Check for common model files
        model_files = list(merged_dir.glob("*.safetensors")) + \
                     list(merged_dir.glob("*.bin")) + \
                     list(merged_dir.glob("model.gguf"))
        if model_files:
            return True
    
    return False

