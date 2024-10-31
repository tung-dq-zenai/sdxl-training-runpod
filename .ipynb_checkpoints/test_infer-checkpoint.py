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


handler = TokenEmbeddingsHandler(
    [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
)

handler.load_embeddings("/workspace/training_sdxl_pti/out_astria/embeddings.pti")

pipe = StableDiffusionXLPipeline(vae = vae , text_encoder = text_encoder_one , text_encoder_2 = text_encoder_two , unet = unet , tokenizer = tokenizer_one ,  tokenizer_2 = tokenizer_two , scheduler = noise_scheduler ).to('cuda')
pipe.load_lora_weights('/workspace/training_sdxl_pti/out_astria/lora.safetensors' , adapter_name = 'lora')
pipe.set_adapters("lora")

prompt = "a <s0><s1> woman doing sexy pose, naked, front view, luxurious bedroom, a window, snowy forest, highres, prominent grain, muted low grain."
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
sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1.0}
results = pipe(**common_args, **sdxl_kwargs).images
for i , img in enumerate(results):
    img.save(f'{i}.png')

