#!/usr/bin/env python3
"""
Patch Axolotl to work with CPU/Apple Silicon.
Fixes the CUDA check issue when no GPU is available.
"""

import sys
import os

# Add axolotl to path
axolotl_path = None
for path in sys.path:
    if 'axolotl' in path and 'site-packages' in path:
        axolotl_path = path
        break

if axolotl_path:
    sys.path.insert(0, axolotl_path)

try:
    import axolotl.cli.config as config_module
    import torch
    
    # Save original function
    original_compute_supports_fp8 = config_module.compute_supports_fp8
    
    # Create patched version
    def patched_compute_supports_fp8():
        """Patched version that handles CPU-only systems."""
        try:
            if not torch.cuda.is_available():
                return False
            return original_compute_supports_fp8()
        except (AssertionError, RuntimeError):
            return False
    
    # Apply patch
    config_module.compute_supports_fp8 = patched_compute_supports_fp8
    print("✅ Patched Axolotl config module for CPU support", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Could not patch Axolotl: {e}", file=sys.stderr)

