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
    for model_id in tqdm(model_names):
        orig_author, orig_model_name = model_id.split('/')
        model_id = f'TornikeO/{orig_model_name}-fp16'

        model = DiffusionPipeline.from_pretrained(
            model_id,
            revision='fp16',
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        print(model)
        args = argparse.Namespace()
        args.mixed_precision = 'fp16'
        args.gradient_accumulation_steps = 1
        args.with_prior_preservation = True
        args.class_data_dir = 'output_images'
        args.num_class_images = 200
        args.sample_batch_size = 4
        args.class_prompt = 'photo of person'
        args.pretrained_model_name_or_path = model_id
        args.revision = 'fp16'
        generate_images.main(args)


        # if Path(output_dir).exists():
        #     import shutil
        #     shutil.rmtree(Path(output_dir))
        
        # api = HfApi()
        # try:
        #     api.delete_repo(new_model_id)
        # except Exception as e:
        #     print(e)

        # remote_repo = api.create_repo(
        #     repo_id=new_model_id,
        #     exist_ok=True
        # )

        # local_repo = Repository(
        #     output_dir, 
        #     revision='fp16',
        #     clone_from=remote_repo,
        #     skip_lfs_files=True,
        # )

        # Path(output_dir).mkdir(exist_ok=True)
        # model.save_pretrained(
        #     output_dir,
        # )   
        # local_repo.push_to_hub(commit_message="Add fp16 files", blocking=True,clean_ok=False, auto_lfs_prune=True)

        # # Clean up afterwards
        
        # shutil.rmtree(output_dir, ignore_errors=True)

    print("Done!")

if __name__ == "__main__":
    main()