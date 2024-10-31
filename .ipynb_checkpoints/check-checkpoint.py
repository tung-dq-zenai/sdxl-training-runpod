from safetensors import safe_open
from safetensors.torch import save_file
import torch
from typing import Dict, Union, Optional
from pathlib import Path

def convert_lora_weights(
    old_weights_path: Union[str, Path],
    new_weights_path: Union[str, Path],
    device: Union[str, int, torch.device] = "cpu"
) -> None:
    """
    Convert old format LoRA weights to new format using safetensors, adding 'unet.' prefix.
    
    Args:
        old_weights_path: Path to the old format LoRA safetensors weights
        new_weights_path: Path to save the converted weights
        device: Device to load the tensors to ("cpu", "cuda", 0, 1, etc.)
    """
    # Convert device to proper format
    if isinstance(device, int):
        device = f"cuda:{device}" if device >= 0 else "cpu"
    
    # Load old weights using safetensors
    old_state_dict = {}
    with safe_open(old_weights_path, framework="pt", device=device) as f:
        for key in f.keys():
            old_state_dict[key] = f.get_tensor(key)
            print(key)
            print(f.get_tensor(key))
    
    # Create new state dict with 'unet.' prefix
    new_state_dict = {
        f'unet.{module_name}': params 
        for module_name, params in old_state_dict.items()
    }
    
    # Save the new state dict using safetensors
    save_file(new_state_dict, new_weights_path)
    
    # Print conversion summary
    print(f"Converted {len(old_state_dict)} LoRA weights from old format to new format")
    print(f"\nExample conversions:")
    # Show first 3 key conversions as examples
    for old_key, new_key in list(zip(old_state_dict.keys(), new_state_dict.keys()))[:3]:
        print(f"  {old_key} -> {new_key}")
    
    print(f"\nSaved converted weights to {new_weights_path}")
    
def validate_conversion(
    old_path: Union[str, Path],
    new_path: Union[str, Path],
    device: Union[str, int, torch.device] = "cpu"
) -> bool:
    """
    Validate that the conversion was successful by comparing tensor values.
    
    Args:
        old_path: Path to the original LoRA weights
        new_path: Path to the converted LoRA weights
        device: Device to load the tensors to
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Load both files
    old_tensors = {}
    new_tensors = {}
    
    with safe_open(old_path, framework="pt", device=device) as f:
        for key in f.keys():
            old_tensors[key] = f.get_tensor(key)
            
    with safe_open(new_path, framework="pt", device=device) as f:
        for key in f.keys():
            # Remove 'unet.' prefix for comparison
            original_key = key.replace('unet.', '', 1)
            new_tensors[original_key] = f.get_tensor(key)
    
    # Validate
    if set(old_tensors.keys()) != set(new_tensors.keys()):
        print("❌ Validation failed: Key mismatch")
        return False
        
    all_match = True
    for key in old_tensors:
        if not torch.equal(old_tensors[key], new_tensors[key]):
            print(f"❌ Validation failed: Tensor mismatch for {key}")
            all_match = False
            
    if all_match:
        print("✅ Validation passed: All tensors match")
    return all_match

# Example usage
if __name__ == "__main__":
    try:
        # Convert weights
        convert_lora_weights(
            old_weights_path="/workspace/training_sdxl_pti/pytorch_lora_weights.safetensors",
            new_weights_path="new_lora.safetensors",
            device="cpu"
        )
        
        # Validate conversion
        validate_conversion(
            old_path="/workspace/training_sdxl_pti/checkpoint/unet/checkpoint-500.lora.safetensors",
            new_path="new_lora.safetensors",
            device="cpu"
        )
        
    except FileNotFoundError:
        print("Error: Safetensors file not found")
    except Exception as e:
        print(f"Unexpected error: {e}")