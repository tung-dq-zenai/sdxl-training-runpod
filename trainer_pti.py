# Bootstrapped from Huggingface diffuser's code.
import fnmatch
import json
import math
import os
import shutil
from typing import List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0 , AttnProcessor
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm.auto import tqdm
from convert_utils import modify_and_save_lora_model , modify_and_save_embedding_model
from dataset_and_utils import (
    PreprocessedDataset,
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict,
)

def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f"failed to is_belong_to_block, due to: {e}")
def main(
    pretrained_model_name_or_path: Optional[
        str
    ] = "./cache",  # "stabilityai/stable-diffusion-xl-base-1.0",
    revision: Optional[str] = None,
    instance_data_dir: Optional[str] = "./dataset/zeke/captions.csv",
    output_dir: str = "ft_masked_coke",
    seed: Optional[int] = 42,
    resolution: int = 512,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 1,
    do_cache: bool = True,
    num_train_epochs: int = 600,
    max_train_steps: Optional[int] = None,
    checkpointing_steps: int = 500000,  # default to no checkpoints
    gradient_accumulation_steps: int = 1,  # todo
    unet_learning_rate: float = 1e-5,
    ti_lr: float = 3e-4,
    lora_lr: float = 1e-4,
    pivot_halfway: bool = True,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,  # todo with tests
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "bf16",
    device: str = "cuda:0",
    token_dict: dict = {"TOKEN": "<s0>"},
    inserting_list_tokens: List[str] = ["<s0>"],
    verbose: bool = True,
    is_lora: bool = True,
    is_sdxl: bool = True,
    lora_rank: int = 32,
) -> None:
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not seed:
        seed = np.random.randint(0, 2**32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if scale_lr:
        unet_learning_rate = (
            unet_learning_rate * gradient_accumulation_steps * train_batch_size
        )

    (
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(pretrained_model_name_or_path, revision, device, weight_dtype, is_sdxl)

    print("# PTI : Loaded models")

    # Initialize new tokens for training.
    if is_sdxl:
        embedding_handler = TokenEmbeddingsHandler(
            [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
        )
    else:
        embedding_handler = TokenEmbeddingsHandler(
            [text_encoder_one], [tokenizer_one]
        )
    embedding_handler.initialize_new_tokens(inserting_toks=inserting_list_tokens)

    text_encoders = [text_encoder_one, text_encoder_two] if is_sdxl else [text_encoder_one]

    unet_param_to_optimize = []
    # fine tune only attn weights

    text_encoder_parameters = []
    for text_encoder in text_encoders:
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                param.requires_grad = True
                print(name)
                text_encoder_parameters.append(param)
            else:
                param.requires_grad = False

    if not is_lora:
        WHITELIST_PATTERNS = [
            # "*.attn*.weight",
            # "*ff*.weight",
            "*"
        ]  # TODO : make this a parameter
        BLACKLIST_PATTERNS = ["*.norm*.weight", "*time*"]

        unet_param_to_optimize_names = []
        for name, param in unet.named_parameters():
            if any(
                fnmatch.fnmatch(name, pattern) for pattern in WHITELIST_PATTERNS
            ) and not any(
                fnmatch.fnmatch(name, pattern) for pattern in BLACKLIST_PATTERNS
            ):
                param.requires_grad_(True)
                unet_param_to_optimize_names.append(name)
                print(f"Training: {name}")
            else:
                param.requires_grad_(False)

        # Optimizer creation
        params_to_optimize = [
            {
                "params": unet_param_to_optimize,
                "lr": unet_learning_rate,
            },
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": 1e-3,
            },
        ]

    else:
        # Do lora-training instead.
        unet.requires_grad_(True)

        from peft import LoraConfig
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        unet_lora_config = LoraConfig(
            r=lora_rank,
            use_dora=False,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet.add_adapter(unet_lora_config)
        unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
        params_to_optimize = [
            {
                "params": unet_lora_parameters,
                "lr": lora_lr,
            },
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": 1e-3,
            },
        ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        weight_decay=1e-4,
    )

    print(f"# PTI : Loading dataset, do_cache {do_cache}")

    train_dataset = PreprocessedDataset(
        instance_data_dir,
        tokenizer_one,
        tokenizer_two,
        vae.float(),
        do_cache=True,
        substitute_caption_map=token_dict,
        is_sdxl=is_sdxl,
    )

    print("# PTI : Loaded dataset")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * gradient_accumulation_steps

    if verbose:
        print(f"# PTI :  Running training ")
        print(f"# PTI :  Num examples = {len(train_dataset)}")
        print(f"# PTI :  Num batches each epoch = {len(train_dataloader)}")
        print(f"# PTI :  Num Epochs = {num_train_epochs}")
        print(f"# PTI :  Instantaneous batch size per device = {train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"# PTI :  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"# PTI :  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps))
    checkpoint_dir = "checkpoint"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    os.makedirs(f"{checkpoint_dir}/unet", exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/embeddings", exist_ok=True)

    for epoch in range(first_epoch, num_train_epochs):
        if pivot_halfway:
            if epoch == num_train_epochs // 2:
                print("# PTI :  Pivot halfway")
                # remove text encoder parameters from optimizer
                params_to_optimize = params_to_optimize[:1]
                optimizer = torch.optim.AdamW(
                    params_to_optimize,
                    weight_decay=1e-4,
                )

        unet.train()
        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")
            global_step += 1
            
            if is_sdxl:
                (tok1, tok2), vae_latent, mask = batch
            else:
                tok1, vae_latent, mask = batch
            vae_latent = vae_latent.to(weight_dtype)

            # tokens to text embeds
            added_kw = None
            if is_sdxl:
                prompt_embeds_list = []
                for tok, text_encoder in zip((tok1, tok2), text_encoders):
                    prompt_embeds_out = text_encoder(
                        tok.to(text_encoder.device),
                        output_hidden_states=True,
                    )

                    pooled_prompt_embeds = prompt_embeds_out[0]
                    prompt_embeds = prompt_embeds_out.hidden_states[-2]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

                # Create Spatial-dimensional conditions.

                original_size = (resolution, resolution)
                target_size = (resolution, resolution)
                crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])

                add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype).repeat(
                    bs_embed, 1
                )

                added_kw = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            else:
                prompt_embeds = text_encoder_one(
                    tok1.to(text_encoder_one.device),
                )[0]

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(vae_latent)
            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            )
            timesteps = timesteps.long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_kw,
            ).sample

            loss = (model_pred - noise).pow(2) * mask
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # every step, we reset the embeddings to the original embeddings.

            for idx, text_encoder in enumerate(text_encoders):
                embedding_handler.retract_embeddings()

            if global_step % checkpointing_steps == 0:
                # save the required params of unet with safetensor

                if not is_lora:
                    tensors = {
                        name: param
                        for name, param in unet.named_parameters()
                        if name in unet_param_to_optimize_names
                    }
                    save_file(
                        tensors,
                        f"{checkpoint_dir}/unet/checkpoint-{global_step}.unet.safetensors",
                    )

                else:
                    unet.save_attn_procs(f"{checkpoint_dir}/unet", weight_name=f"checkpoint-{global_step}.lora.safetensors")

                embedding_handler.save_embeddings(
                    f"{checkpoint_dir}/embeddings/checkpoint-{global_step}.pti",
                )

    # final_save
    print("Saving final model for return")
    if not is_lora:
        tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize_names
        }
        save_file(
            tensors,
            f"{output_dir}/unet.safetensors",
        )
    else:
        unet.save_attn_procs(output_dir, weight_name="lora.safetensors")
        lora_file = f"{output_dir}/lora.safetensors"
        modify_and_save_lora_model(input_model_path = lora_file , output_model_path =lora_file )

    embedding_file = f"{output_dir}/embeddings.pti"
    new_embedding_file = f"{output_dir}/embeddings.pt"
    embedding_handler.save_embeddings(embedding_file)
    modify_and_save_embedding_model(input_model_path = embedding_file , output_model_path = new_embedding_file)
    os.remove(embedding_file)
    to_save = token_dict
    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(to_save, f)


if __name__ == "__main__":
    main()
