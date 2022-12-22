import argparse
import hashlib
import inspect
import itertools
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from accelerate import logging
from requests import HTTPError
import numpy as np


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")




# LOG = logging.getLogger(__name__)
# if __name__ == '__main__':
# logging.basicConfig(level=logging.INFO)

# with logging_redirect_tqdm():
#     for i in trange(9):
#         if i == 4:
#             LOG.info("console logging redirected to `tqdm.write()`")
import time
# with tqdm_logging_redirect(tqdm_class=tqdm, loggers=[LOG]):
#     for i in trange(9):
#         # if i == 4:
#         LOG.info("console logging redirected to `tqdm.write()`")
    # logging restored
logger = get_logger(__name__)
# logger.basicConfig(filename='logs.txt',
#                 filemode='a',
#                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                 datefmt='%H:%M:%S', 
#                 level=logging.DEBUG, force=True)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def get_parser():

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="multimodalart/sd-fine-tunable",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="instance_data_dir",
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="class_data_dir",
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="photo of tornikeo",
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="photo of a person",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        choices=range(0, 400),
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt. Recommended to be 25 times the number of instance imags."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", 
        default=False,
        action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder",
        default=False,
        action="store_true", help="Whether to train the text encoder. **Requires at least 24GB VRAM to not crash**.")
    parser.add_argument(
        "--train_batch_size", type=int, 
        default=1, help="Batch size (per device) for the training dataloader. With `train_text_encoder`, a single batch requires 18GB of VRAM."
    )
    parser.add_argument(
        "--sample_batch_size", 
        type=int,
        default=7,
        choices=range(1,24),
        help="Batch size (per device) for sampling images. This is for class image *generation*, much more lightweight than training."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=400,
        choices=range(1,2000),
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs. **Recommended to be 100 times the number of input images**.",
    )
    parser.add_argument("--save_steps", type=int, default=9999, help="Save checkpoint every X updates steps. -1 disables the feature.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        choices=range(1,16),
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        choices=np.linspace(5e-7,5e-5,100),
        help="Initial learning rate (after the potential warmup period) to use. Decrease to 5e-7 if you see noisy generated images.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size. Only useful for multi-GPU training.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help=(
            'The scheduler type to use.'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", default=True, help="Whether or not to use 8-bit Adam from bitsandbytes. Makes it possible to train with smaller GPUs."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", default=False, action="store_true", help="Whether or not to push the model to the Hub. If you use this, make sure to also specify `hub_token` and ")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument('--do_not_train', default=False, action="store_true", help="Useful for development only.")
    parser.add_argument('--do_not_save_pretrained', default=False, action="store_true", help="Useful for development only.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config. **",
            "**fp16 is necessary for smaller GPU training."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    return parser
    
def parse_args(input_args=None):
    parser = get_parser()
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            for attempts in range(3):
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    break
                except Exception as e:
                    print("Some weird huggingface loading error occured. Waiting and retrying once more. Error: ", e)
                    time.sleep(1)
                    if attempts == 2: raise e
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            # pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     import huggingface_hub
    #     if args.push_to_hub:
    #         assert args.hub_token is not None
    #         huggingface_hub.login(args.hub_token)
    #         if args.hub_model_id is None:
    #             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
    #         else:
    #             repo_name = args.hub_model_id
    #         repo = Repository(args.output_dir, clone_from=repo_name)
            

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    

    # if accelerator.is_main_process:
    #     logger.info("Saving the model...")
    #     if not args.do_not_save_pretrained:
    #         pipeline = DiffusionPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path,
    #             unet=accelerator.unwrap_model(unet),
    #             text_encoder=accelerator.unwrap_model(text_encoder),
    #             revision='fp16',
    #             torch_dtype=torch.float16,
    #         )
    #         pipeline.save_pretrained(args.output_dir)

    #     if args.push_to_hub:
    #         logger.info(f"Pushing the model to the hub (repo-name {repo_name}) ...")
    #         repo.push_to_hub(commit_message="End of training", blocking=True, clean_ok=False, auto_lfs_prune=True)

    # accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
