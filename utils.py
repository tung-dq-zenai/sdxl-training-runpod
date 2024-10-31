import os
import shutil
import subprocess


def prepare_training_data(aws_link_list, data_path):

    for i, aws_link in enumerate(aws_link_list):
        image_filename = os.path.join(data_path, f"{i}.png")
        subprocess.run(
            ["curl", "-o", image_filename, aws_link],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def handle_data_paths(
    instance_data,
):

    # Create dir name compatible to Kohya scripts
    instance_dir_name = f"data"

    for dir_name in [instance_dir_name]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)

    # Download training images to training data path
    prepare_training_data(
        instance_data, instance_dir_name
    )

    return instance_dir_name