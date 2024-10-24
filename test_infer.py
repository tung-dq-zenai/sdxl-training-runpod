from dataset_and_utils import (
    PreprocessedDataset,
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict,
)
import torch
from diffusers import StableDiffusionXLPipeline
import os
from safetensors.torch import load_file
import json
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

pretrained_model_name_or_path = "BKM1804/lustify-sdxl"
local_weights_cache = 'out_astria'
revision = None
weight_dtype = torch.float32
is_sdxl = True
device = "cuda"
(
    tokenizer_one,
    tokenizer_two,
    noise_scheduler,
    text_encoder_one,
    text_encoder_two,
    vae,
    unet,
) = load_models(pretrained_model_name_or_path, revision, device, weight_dtype, is_sdxl)

# # tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))
# # tensors = load_file("/workspace/training_sdxl_pti/checkpoint/unet/checkpoint-500.lora.safetensors")


# unet_lora_attn_procs = {}
# name_rank_map = {}
# for tk, tv in tensors.items():
#     # up is N, d
#     if tk.endswith("up.weight"):
#         proc_name = ".".join(tk.split(".")[:-3])
#         r = tv.shape[1]
#         name_rank_map[proc_name] = r

# for name, attn_processor in unet.attn_processors.items():
#     cross_attention_dim = (
#         None
#         if name.endswith("attn1.processor")
#         else unet.config.cross_attention_dim
#     )
#     if name.startswith("mid_block"):
#         hidden_size = unet.config.block_out_channels[-1]
#     elif name.startswith("up_blocks"):
#         block_id = int(name[len("up_blocks.")])
#         hidden_size = list(reversed(unet.config.block_out_channels))[
#             block_id
#         ]
#     elif name.startswith("down_blocks"):
#         block_id = int(name[len("down_blocks.")])
#         hidden_size = unet.config.block_out_channels[block_id]

#     module = LoRAAttnProcessor2_0(
#         hidden_size=hidden_size,
#         cross_attention_dim=cross_attention_dim,
#         rank=name_rank_map[name],
#     )
#     unet_lora_attn_procs[name] = module.to("cuda")

# unet.set_attn_processor(unet_lora_attn_procs)
# unet.load_state_dict(tensors, strict=False)

handler = TokenEmbeddingsHandler(
    [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
)
# handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))
handler.load_embeddings("/workspace/training_sdxl_pti/out_astria/embeddings.pti")

pipe = StableDiffusionXLPipeline(vae = vae , text_encoder = text_encoder_one , text_encoder_2 = text_encoder_two , unet = unet , tokenizer = tokenizer_one ,  tokenizer_2 = tokenizer_two , scheduler = noise_scheduler ).to('cuda')
pipe.load_lora_weights('/workspace/training_sdxl_pti/ckpt/pytorch_lora_weights.safetensors' , adapter_name = 'lora')
pipe.set_adapters("lora")

prompt = "a <s0><s1> woman doing sexy pose, naked, front view, luxurious bedroom, a window, snowy forest, highres, prominent grain, muted low grain,"
# for k, v in token_map.items():
#     prompt = prompt.replace(k, v)
# print(f"Prompt: {prompt}")
neg_prompt = "(worst quality, low quality, normal quality), disabled body, sketches, (manicure:1.2), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, extra fingers, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, momochrome, (ugly) , bad hand ,  bad leg , lost fingers, 4 fingers."
num_outputs = 4
width = 768
height = 1024
guidance_scale = 3.5
num_inference_steps = 40
seed = 1234
lora_scale = 0.7
generator = [
    torch.Generator(device="cuda").manual_seed(seed + i)
    for i in range(num_outputs)
]
sdxl_kwargs = {}
common_args = {
    "prompt": [prompt] * num_outputs if prompt is not None else None,
    "negative_prompt": [neg_prompt] * num_outputs if prompt is not None else None,
    "guidance_scale": guidance_scale,
    "generator": generator,
    "num_inference_steps": num_inference_steps,
    "max_sequence_length":256
}            
sdxl_kwargs["width"] = width
sdxl_kwargs["height"] = height
sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1}
results = pipe(**common_args, **sdxl_kwargs).images
for i , img in enumerate(results):
    img.save(f'{i}.png')

