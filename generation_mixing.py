import argparse
import os
import sys
import glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from utils import get_folder_names_and_indexes

from models.sd.ldm.util import instantiate_from_config
from models.sd.ldm.models.diffusion.ddim_mixing import DDIMSampler
from models.sd.ldm.models.diffusion.plms_mixing import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


import clip

class embeddingPrompt():
    def __init__(self):
        self.total_combination=1 #defaul 1 due to 1 attribute being studied
        self.device=torch.device("cuda")
        self.setup_clip_model()       
        self.clip_layers_num = self.clip_model.transformer.layers
        
        self.tokenized_text_queries=torch.vstack([clip.tokenize("A headshot of a person").to(self.device) for i in range(2)])
        
        
    def setup_clip_model(self):
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        

    def tokenize_embed(self,prompt,dim):
        tokenized_text_queries=torch.vstack([clip.tokenize(prompt).to(self.device) for i in range(dim)])
        xp = self.clip_model.token_embedding(tokenized_text_queries).type(self.clip_model.dtype)
        text_queries_embedding = xp + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        
        return text_queries_embedding

    def construct_fair_text_features(self, text_queries_embedding, fairtoken_model,last_word_idx=6, token_length=3, attributesLen=1):
         """
         insert fair_tokens to the original text queries' embeddings and obtain the
         corresponding text feature
         :return:
         """
         
         def indexFinder(attributesLen):
            if attributesLen == 2:
                return [[torch.tensor([0, 2]), torch.tensor([1, 3])], [torch.tensor([0, 1]), torch.tensor([2, 3])]]
            elif attributesLen == 1:
                return [[torch.tensor([0, 1])]]
            
         indexOrd = indexFinder(attributesLen)   


         x = text_queries_embedding.detach() # (108, 77, 768)
         # add FairToken to the corresponding place
         # for 1st attr, replace from the last word index; for 2nd and later, add to them
         for i, each_index in enumerate(indexOrd):
             for index in each_index:
                 if i == 0:
                     x[index, last_word_idx:last_word_idx + token_length, :] = (
                     fairtoken_model[i])
                 else:
                     x[index, last_word_idx:last_word_idx + token_length, :] += (
                     fairtoken_model[i])
         return x
        
    def CLIP_embed(self,x):
         x = x.permute(1, 0, 2)
         for ll in range(self.clip_layers_num):
             x = self.clip_model.transformer.resblocks[ll](x)
         x = x.permute(1, 0, 2)
         return self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (108, 77, 768)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


# # Safety filter often flags normal images (disable temporarily)
# def check_safety(x_image):
#     return x_image, 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a headshot of a person",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default='./ckpts/test'
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        default=True,  # Edit: added
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=10,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=6,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='models/sd/configs/stable-diffusion/v1-inference.yaml',  # Edit: added
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='models/sd/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt',  # Edit: added
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full"
    )

    # added parameters
    parser.add_argument(
        '--attr-list',
        type=str,
        default="High_Cheekbones",  # Edit: added
        help='input the attributes that you want to debias, separated by commas. Eg, Pale_Skin,Eyeglasses,...'

    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=5,
        help='gpu number'
    )
    parser.add_argument(
        '--prompt-path',
        type=str,
        default='./ckpts/P/a_headshot_of_a_person_High_Cheekbones/original_prompt_embedding/basis_final_embed_29.pt',  # Edited: added
        help='checkpoint of the learned token embeddings that are used for image generation in Stable Diffusion'
    )
    
    #Prompt Queuing
    parser.add_argument(
        '--ts',
        type=int,
        default=40,
        help='DDIM_step to set as Transition step for Prompt Queuing'
    )
    #Attention Amplication
    parser.add_argument(
        '--aa',
        type=int,
        default=10,
        help='Attention Amplication scale, c'
    )
    #Token to Amplify
    parser.add_argument(
        '--st',
        action='append_const',
        const=int,
        default=[9],
        help='Selected tokens to Attention Amplify'
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    emb = torch.load(opt.prompt_path).to(device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    # Get combination
    folder_with_indexes = get_folder_names_and_indexes(
        opt.attr_list.split(','))
    for folder, index in folder_with_indexes.items():

        sample_path = os.path.join(opt.outdir, folder)
        os.makedirs(sample_path, exist_ok=True)

        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn(
                [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


        #(Base Prompt)
        eb=embeddingPrompt()
        with torch.no_grad():
            text_queries_embedding=eb.tokenize_embed(opt.prompt, len(emb))
            BasePrompt_Clip_embedding=eb.CLIP_embed(text_queries_embedding)
        
        
        
        
        
        data = [[torch.stack([BasePrompt_Clip_embedding[index],emb[index]]) for i in range(batch_size)]]

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()

                    seed_everything(opt.seed)

                    for n in range(opt.n_iter):

                        # The prompt embedding for cross-attention in the Stable Diffusion
                        for c in data:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(
                                    batch_size * [""])

                            # Edited
                            c = torch.vstack(c)

                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, tmp = sampler.sample(S=opt.ddim_steps,
                                                               conditioning=c,
                                                               batch_size=opt.n_samples,
                                                               shape=shape,
                                                               verbose=False,
                                                               unconditional_guidance_scale=opt.scale,
                                                               unconditional_conditioning=uc,
                                                               eta=opt.ddim_eta,
                                                               x_T=start_code,
                                                               transitionS=opt.ts,
                                                               aa=opt.aa,
                                                               selectedS=opt.st)

                            x_samples_ddim = model.decode_first_stage(
                                samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = check_safety(
                                x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(
                                x_samples_ddim).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * \
                                        rearrange(
                                            x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(
                                        x_sample.astype(np.uint8))
                                    img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(
                                        sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * \
                            rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        # img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        img.save(os.path.join(outpath, f'grid_{folder}.png'))
                        grid_count += 1

                    toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
