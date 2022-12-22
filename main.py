from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from huggingface_hub import HfFolder, Repository, whoami
from huggingface_hub.hf_api import HfApi
import huggingface_hub
from typing import Optional, List
import argparse
import os
from tqdm import tqdm
import shutil
import generate_images
import subprocess
import accelerate
from accelerate.commands import launch
from slugify import slugify
import pickle

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main():
    assert 'HF_AUTH_TOKEN' in os.environ
    token = os.environ['HF_AUTH_TOKEN']
    huggingface_hub.login(token)

    output_dir = 'images'

    model_names = [
        "nitrosocke/Ghibli-Diffusion",
        "nitrosocke/redshift-diffusion",
        "nitrosocke/Nitro-Diffusion",
        "nitrosocke/Future-Diffusion",
        "nitrosocke/Arcane-Diffusion",
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
        "multimodalart/sd-fine-tunable",
        "Linaqruf/anything-v3.0",
        "DGSpitzer/Cyberpunk-Anime-Diffusion",
        "wavymulder/Analog-Diffusion",
        "dallinmackay/Van-Gogh-diffusion",
        "dreamlike-art/dreamlike-diffusion-1.0"
    ]

    api = HfApi()

    remote_repo = api.create_repo(
        repo_id='TornikeO/dreambooth-class-img-cache',
        exist_ok=True
    )

    repo_directory = Path('repo')

    repo_directory.mkdir(exist_ok=True)
    results_dir = Path('results')
    repo_dir = Path('repo')
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(exist_ok=True)
    local_repo = Repository(
        repo_dir, 
        clone_from=remote_repo,
        skip_lfs_files=False,
    )

    for model_id in tqdm(model_names):
        orig_author, orig_model_name = model_id.split('/')
        model_id = f'TornikeO/{orig_model_name}-fp16'
        args = argparse.Namespace()
        args.mixed_precision = 'fp16'
        args.gradient_accumulation_steps = 1
        args.with_prior_preservation = True
        args.num_class_images = 200

        args.sample_batch_size = 30

        args.class_prompt = 'photo of person'
        args.pretrained_model_name_or_path = model_id
        args.revision = 'fp16'
        args.class_data_dir = results_dir / slugify(args.pretrained_model_name_or_path) / slugify(args.class_prompt)
        
        args.sample_batch_size = 70
        try:
            generate_images.main(args)
        except Exception as e:
            args.sample_batch_size = 30
            generate_images.main(args)
        pickle.dump(args, (args.class_data_dir.parent / 'args.pickle').open('wb'))
        shutil.copytree(results_dir, repo_dir, dirs_exist_ok=True)
        local_repo.push_to_hub(commit_message="Add cache files", blocking=False, clean_ok=True, auto_lfs_prune=True)
    print("Done!")

if __name__ == "__main__":
    main()