from train import train as train_pti
import os
import sys
from convert_lora import modify_and_save_lora_model
os.environ['DISABLE_AUTO_CAPTIONS'] = 'true' 
TRAINING_MODEL_DIR = "out_astria"
train_pti(
    input_images=f"/workspace/training_sets/renca_training_images/",
    output_dir=TRAINING_MODEL_DIR,
    pretrained_model_name_or_path="BKM1804/lustify-sdxl",
    seed= 1337,
    resolution=768,
    train_batch_size= 4,
    num_train_epochs= 4000,
    max_train_steps=10,
    is_lora=True,
    is_sdxl=True,
    unet_learning_rate= 2e-6,
    ti_lr= 3e-4,
    lora_lr= 1e-4,
    lora_rank=32,
    class_name="woman",
    token_string="ohwx",
    caption_prefix= 'a photo of',
    checkpointing_steps=100000,
    # Uses CLIPSEG to mask target in the loss function
    # mask_target_prompts=tune.name if use_photo and (tune.mask_target or os.environ.get('MASK_TARGET')) else None,
    mask_target_prompts = None,
    use_face_detection_instead = False
    # *(['--use_face_detection_instead', 'USE_FACE_DETECTION_INSTEAD'] if tune.face_crop and tune.name in "man woman boy girl male female" else []),
)