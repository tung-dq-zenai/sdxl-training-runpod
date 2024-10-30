import argparse
import os
import shutil
from preprocess import preprocess
from trainer_pti import main



def train(
    input_images,
    output_dir,
    pretrained_model_name_or_path,
    seed=None,
    resolution=512,
    train_batch_size=4,
    num_train_epochs=4000,
    max_train_steps=1000,
    is_lora=True,
    is_sdxl=False,
    unet_learning_rate=1e-6,
    ti_lr=3e-4,
    lora_lr=1e-4,
    lora_rank=32,
    lr_scheduler="constant",
    lr_warmup_steps=100,
    token_string="sks",
    class_name="",
    caption_prefix="a photo of",
    mask_target_prompts=None,
    crop_based_on_salience=True,
    use_face_detection_instead=False,
    clipseg_temperature=1.0,
    verbose=True,
    checkpointing_steps=200
):
    # Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
    if os.environ.get("NUM_TOKENS"):
        token_map = token_string + ":" + os.environ.get("NUM_TOKENS")
    else:
        token_map = token_string + ":2"

    # Process 'token_to_train' and 'input_data_tar_or_zip'
    inserting_list_tokens = token_map.split(",")

    token_dict = {}
    running_tok_cnt = 0
    all_token_lists = []
    for token in inserting_list_tokens:
        n_tok = int(token.split(":")[1])

        token_dict[token.split(":")[0]] = "".join(
            [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
        )
        all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])

        running_tok_cnt += n_tok

    input_dir = preprocess(
        input_path=input_images,
        output_dir=output_dir,
        caption_text=caption_prefix,
        token_string=token_string,
        class_name=class_name,
        mask_target_prompts=mask_target_prompts,
        target_size=resolution,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=clipseg_temperature,
        substitution_tokens=list(token_dict.keys()),
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    main(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        instance_data_dir=os.path.join(input_dir, "captions.csv"),
        output_dir=output_dir,
        seed=seed,
        resolution=resolution,
        train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=1,
        unet_learning_rate=unet_learning_rate,
        ti_lr=ti_lr,
        lora_lr=lora_lr,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        token_dict=token_dict,
        inserting_list_tokens=all_token_lists,
        verbose=verbose,
        checkpointing_steps=checkpointing_steps,
        scale_lr=False,
        max_grad_norm=1.0,
        allow_tf32=True,
        mixed_precision="bf16",
        device="cuda:0",
        lora_rank=lora_rank,
        is_lora=is_lora,
        is_sdxl=is_sdxl,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with given parameters.')

    parser.add_argument('--input_images', type=str, required=True, help='Input images folder.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=4000)
    parser.add_argument('--max_train_steps', type=int, default=1000)
    parser.add_argument('--is_lora', action='store_false')
    parser.add_argument('--is_sdxl', action='store_true')
    parser.add_argument('--unet_learning_rate', type=float, default=1e-6)
    parser.add_argument('--ti_lr', type=float, default=3e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lr_scheduler', type=str, default="constant", choices=['constant', 'linear'])
    parser.add_argument('--lr_warmup_steps', type=int, default=100)
    parser.add_argument('--token_string', type=str, default="sks")
    parser.add_argument('--caption_prefix', type=str, default="a photo of")
    parser.add_argument('--class_name', type=str, default="")
    parser.add_argument('--mask_target_prompts', type=str, default=None)
    parser.add_argument('--crop_based_on_salience', type=bool, default=True)
    parser.add_argument('--use_face_detection_instead', type=bool, default=False)
    parser.add_argument('--clipseg_temperature', type=float, default=1.0)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--checkpointing_steps', type=int, default=20000)

    args = parser.parse_args()
    print(args)

    train(
        input_images=args.input_images,
        output_dir=args.output_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        is_lora=args.is_lora,
        is_sdxl=args.is_sdxl,
        unet_learning_rate=args.unet_learning_rate,
        ti_lr=args.ti_lr,
        lora_lr=args.lora_lr,
        lora_rank=args.lora_rank,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        token_string=args.token_string,
        caption_prefix=args.caption_prefix,
        class_name=args.class_name,
        mask_target_prompts=args.mask_target_prompts,
        crop_based_on_salience=args.crop_based_on_salience,
        use_face_detection_instead=args.use_face_detection_instead,
        clipseg_temperature=args.clipseg_temperature,
        verbose=args.verbose,
        checkpointing_steps=args.checkpointing_steps
    )