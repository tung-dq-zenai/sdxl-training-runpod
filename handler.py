''' infer.py for runpod worker '''

import os

import runpod
from runpod.serverless.utils.rp_validator import validate

from schema import TRAIN_SCHEMA

from train import train
from utils import handle_data_paths
from s3_helper import S3Helper

AWS_S3_BUCKET_NAME = "lustylens"
AWS_S3_IMAGES_PATH = "generations"
OUTPUT_FOLDER = "model"
DATA_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

s3_helper = S3Helper()

def run(job):
    '''
    Run training or inference job.
    '''
    job_input = job['input']
    print(f"Job input: {job_input}")
    job_id = job['id']
        
    validate_train = validate(job_input, TRAIN_SCHEMA)
    print(f"Validated input: {validate_train}")
    if 'errors' in validate_train:
        return {'error': validate_train['errors']}
    
    validate_train = validate_train['validated_input']
    handle_data_paths(
        validate_train['images'], DATA_FOLDER
    )
    
    # save source data for face swap
    os.system(f"cp -r {DATA_FOLDER} {OUTPUT_FOLDER}")
    
    train(
        input_images=DATA_FOLDER,
        output_dir=OUTPUT_FOLDER,
        pretrained_model_name_or_path="trongg/lustify_v4",
        seed=42,
        resolution=768,
        train_batch_size=validate_train['batch_size'],
        num_train_epochs=4000,
        max_train_steps=validate_train['steps'],
        is_lora=True,
        is_sdxl=True,
        unet_learning_rate=1e-6,
        ti_lr=3e-4,
        lora_lr=1e-4,
        lora_rank=32,
        lr_scheduler='constant',
        lr_warmup_steps=100,
        token_string=validate_train['identifier'],
        caption_prefix="a photo of",
        class_name=validate_train['class_name'],
        mask_target_prompts=None,
        crop_based_on_salience=True,
        use_face_detection_instead=True,
        clipseg_temperature=1.0,
        verbose=True,
        checkpointing_steps=500,
    )
    
    os.system(f"zip -r {job_id}.zip {OUTPUT_FOLDER}")
    s3_uri = s3_helper.upload(f"{job_id}.zip", AWS_S3_BUCKET_NAME, f"{AWS_S3_IMAGES_PATH}/{job_id}.zip")
    
    os.system(f"rm -rf {OUTPUT_FOLDER}")
    os.system(f"rm -rf {job_id}.zip")
    os.system(f"rm -rf {DATA_FOLDER}")
    
    return {
        "model_url": s3_helper.s3_uri_to_link(s3_uri)
    }


runpod.serverless.start({"handler": run})