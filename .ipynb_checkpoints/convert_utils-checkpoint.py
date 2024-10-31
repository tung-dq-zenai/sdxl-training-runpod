import torch
from safetensors.torch import load_file, save_file

def add_processor_to_keys(keys):
    """Helper function to update keys by adding 'processor' after 'attn*' and before 'to'."""
    updated_keys = []
    for key in keys:
        key_parts = key.split('.')
        new_key_parts = []

        for i, part in enumerate(key_parts):
            if part == '0' and (i > 0 and key_parts[i - 1].startswith('to')) and (i + 1 < len(key_parts) and key_parts[i + 1] == 'lora'):
                continue
            new_key_parts.append(part)
            # Insert 'processor' after 'attn' if followed by 'to*'
            if part.startswith('attn') and (i + 1 < len(key_parts)) and key_parts[i + 1].startswith('to'):
                new_key_parts.append('processor')

        # Join the modified parts back together
        modified_key = '.'.join(new_key_parts)
        modified_key = '.'.join(new_key_parts).replace('lora_A', 'lora.down').replace('lora_B', 'lora.up')
        updated_keys.append(modified_key)

    return updated_keys

def modify_and_save_lora_model(input_model_path, output_model_path):
    model = load_file(input_model_path)

    new_state_dict = {}
    for key, value in model.items():
        updated_key = add_processor_to_keys([key])[0]
        new_state_dict[updated_key] = value

    save_file(new_state_dict, output_model_path)
    print(f"Modified model saved to {output_model_path}")
def modify_and_save_embedding_model(input_model_path , output_model_path):
    state_dict = load_file(input_model_path)
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.shape[1] == 768:
            new_key = "clip_l" 
        elif tensor.shape[1] == 1280:
            new_key = "clip_g"  
        else:
            new_key = key  
        new_state_dict[new_key] = tensor  
    torch.save(new_state_dict, output_model_path)