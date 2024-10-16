from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import functional as F
import os
from torchvision.transforms import transforms
from einops import rearrange
from torchvision.utils import make_grid
import numpy as np
import argparse
import ast

##############################################  1.PREPARATION  ##############################################

# input experimental parameters
desc = "The hyperparameters for cross-attention analysis"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--tSA', type=str, default='Smiling', help='target sensitive attribute')
parser.add_argument('--device', type=str, default='cuda:0', help='e.g.: cuda:0')
parser.add_argument('--prompts_mode', type=str, default='I2H', help='HP, ITI-GEN, H2I, or I2H')
parser.add_argument('--path', type=str, default='', help='path of the ckpts folder')
parser.add_argument('--sample_list', type=str, default=None, help='a list of sample indexes (from 0 to 499), e.g., [0, 1, 2, 3, 4, 5]')
parser.add_argument('--step_start_switch', type=int, default=15, help='the step to switch the prompt for H2I (HP_2_ITI) or I2H (ITI_2_HP) experiments')
args = parser.parse_args()

tSA = args.tSA
device = args.device
prompts_mode = args.prompts_mode
path = args.path
step_start_switch = args.step_start_switch
sample_list = ast.literal_eval(args.sample_list) if args.sample_list else None
target_layer = None
target_head = None

# representative samples for cross-attention analysis
sample_dict = {'Smiling': [1, 30, 71, 86, 92, 97, 100, 240, 340, 478],
               'High_Cheekbones': [140, 145, 198, 218, 350, 366, 394, 444, 489, 497],
               'Gray_Hair': [200, 217, 223, 252, 261, 263, 305, 317, 340, 352],
               'Chubby': [59, 64, 76, 107, 191, 220, 230, 289, 369, 381]}
if sample_list == None:
    sample_list = sample_dict[tSA]

# tSA & Prompts
prompt_mode_dict = {'H2I': {'stage_list': ['HP_2_ITI, Stage 1 (HP)','HP_2_ITI, Stage 2 (ITI)'], 'inference_step_range_list': [[0,step_start_switch],[step_start_switch,51]]},
                    'I2H': {'stage_list': ['ITI_2_HP, Stage 1 (ITI)','ITI_2_HP, Stage 2 (HP)'], 'inference_step_range_list': [[0,step_start_switch],[step_start_switch,51]]},
                    'HP': {'stage_list': range(51), 'inference_step_range_list': [[i, i+1] for i in range(0, 51)]},
                    'ITI-GEN': {'stage_list': range(51), 'inference_step_range_list': [[i, i+1] for i in range(0, 51)]}}
stage_list = prompt_mode_dict[prompts_mode]['stage_list']
inference_step_range_list = prompt_mode_dict[prompts_mode]['inference_step_range_list']

HP_dict = {'Smiling': 'A headshot of a person smiling beaming grinning',
            'High_Cheekbones': 'A headshot of a person high cheek bones',
            'Gray_Hair': 'A headshot of a person with gray hair',
            'Chubby': 'A headshot of a person chubby fleshy obese'}
prompt_HP = HP_dict[tSA]
prompt_ITI_GEN = 'A headshot of a person S0 S1 S2'
ITI_GEN = torch.load(os.path.join(path, f"ckpts/a_headshot_of_a_person_{tSA}/original_prompt_embedding/basis_final_embed_29.pt"))[0, :, :].unsqueeze(0)

# prepare setting file
# this file is used in stable diffusion pipeline and cross-attention map visualization
settings = {'prompt_embeds': {},
            'basePath': f"./%s/{prompts_mode}" % (tSA),
            'stage': {},
            'step_start_switch': step_start_switch,
            'sample_index': sample_list[0]}
voca_list = prompt_HP.split(" ")
settings['vocabulary'] = {'S0': voca_list[-3],
                          'S1': voca_list[-2],
                          'S2': voca_list[-1]}
torch.save(settings, "settings.pt")



##############################################  2.INITIALIZATION  ##############################################

# prepare pipeline and assign parameters
seed = 42
gen = set_seed(seed)
num_infe_steps = 50
model_id = 'compVis/stable-diffusion-v1-4'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# save HP and ITI-GEN embeddings in the setting file
with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        for modes in ['HP_embed_get', 'ITI_embed_get']:
            settings = torch.load('settings.pt')
            settings['mode'] = modes
            torch.save(settings, "settings.pt")
            gen.set_state(torch.load(os.path.join("state", f"state_bf_sample_{sample_list[0]}.pt")))  # note the embedding is invariant across different samples and generation steps
            out = pipe(prompt_HP, num_inference_steps=1, generator=gen,
                       override_text_embeddings=ITI_GEN if modes == 'ITI_embed_get' else None)



##############################################  3.GENERATION  ##############################################

settings = torch.load("settings.pt")
settings['mode'] = prompts_mode

stage_list = []
for i in sample_list:
    settings['sample_index'] = i
    torch.save(settings, "settings.pt")
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            gen.set_state(torch.load(os.path.join("state", f"state_bf_sample_{i}.pt")))
            out = pipe(prompt_ITI_GEN, num_inference_steps=num_infe_steps, generator=gen,
                       override_text_embeddings=ITI_GEN, step_start_switch=step_start_switch)

            stage = torch.load('settings.pt')['stage']
            for inference_step_ranges in inference_step_range_list:
                stage_text = stage[inference_step_ranges[0]]
                if stage_text not in stage_list:
                    stage_list.append(stage_text)
                settings['print_mode'] = stage_text
                torch.save(settings, "settings.pt")
                for word in prompt_ITI_GEN.split(" "):
                    heat_map = tc.compute_global_heat_map(inference_step_range=inference_step_ranges,
                                                          layer_idx=target_layer, head_idx=target_head)
                    heat_map = heat_map.compute_word_heat_map(word)
                    wordstepPath = os.path.join(settings['basePath'], stage_text, word)
                    settings = torch.load('settings.pt')
                    settings['wordstepPath'] = wordstepPath
                    torch.save(settings, "settings.pt")
                    if not os.path.exists(wordstepPath):
                        os.makedirs(wordstepPath)
                    plt.axis('off')
                    imgpath = os.path.join(torch.load('settings.pt')['basePath'], stage_text, 'rawimg_input')
                    img_stepj_input = Image.open(os.path.join(imgpath, f'{i}.png'))
                    heat_map.plot_overlay(img_stepj_input, os.path.join(wordstepPath, f"{i}.png"))

shutil.move("settings.pt", settings['basePath'])



##############################################  4.VISUALIZATION  ##############################################

# helper functions
def resize_and_pad(tensor_image):
    # make sure tensor within [0,1]
    if tensor_image.dtype == torch.uint8:
        tensor_image = tensor_image.float() / 255.0
    
    # transform tensor to PIL
    pil_image = F.to_pil_image(tensor_image)
    
    # change size
    resized_image = pil_image.resize((128, 153))
    output_tensor = F.to_tensor(resized_image)

    return output_tensor


def make_sample(basePath, outFile, sample_list, col=3, treshold=100):

    preprocess = transforms.Compose([transforms.ToTensor()])
    pathdir = [f"{i}.png" for i in sample_list]

    all_samples = []
    for i in tqdm(range(len(pathdir))):
        mini_grid_list = []
        for j in basePath:
            _path = os.path.join(j, pathdir[i])
            image = preprocess(Image.open(_path))

            if image.shape[0] == 3:
                    image = torch.cat((resize_and_pad(image), torch.ones(1, 153, 128)), dim=0)
            mini_grid_list.append(image)

            if len(pathdir) <= treshold:
                all_samples.append(image)

        # Output Mini Grid
        mini_grid = torch.stack(mini_grid_list, 0)
        # print(mini_grid.shape)
        mini_grid = make_grid(mini_grid, nrow=col)
        mini_grid = 255. * rearrange(mini_grid, 'c h w -> h w c').cpu().numpy()
        mini_img = Image.fromarray(mini_grid.astype(np.uint8))
        mini_img.save(outFile.replace("grid.png", f"{sample_list[i]}.png"))

    if len(pathdir) <= treshold:
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = make_grid(grid, nrow=col)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))

    if len(pathdir) <= treshold:
        original_size = img.size
        new_size = (original_size[0] // 2, original_size[1] // 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(outFile)
    else:
        print("Did not output overall grid as it is too big, trying reducing sample size below 100")

# grid generation
Image.MAX_IMAGE_PIXELS = None
ThebasePath = os.path.join(tSA,prompts_mode)
prompt = prompt_ITI_GEN.split(" ")

for i in stage_list:
    basePath = os.path.join(ThebasePath, str(i))
    outFile = os.path.join(basePath, "grid", "grid.png")

    # Check for Grid file directionary
    if not os.path.exists(os.path.join(basePath, "grid")):
        os.makedirs(os.path.join(basePath, "grid"))

    path = []
    path.append(os.path.join(basePath, 'rawimg_input_grid'))

    for word in prompt:
        path.append(os.path.join(basePath, word))

    path.append(os.path.join(basePath, 'rawimg_output_grid'))

    make_sample(path, outFile, sample_list, col=len(path))

# make grids of grid
basePath = os.path.join(ThebasePath, 'grid_master')
if not os.path.exists(basePath):
    os.makedirs(basePath)

outFile = os.path.join(basePath, "grid.png")
path = []
for i in stage_list:
    samplesPath = os.path.join(ThebasePath, str(i), 'grid')
    path.append(samplesPath)

make_sample(path, outFile, sample_list, col=1)

if not os.path.exists('result'):
    os.makedirs('result')

for item in os.listdir(basePath):
    if item != 'grid.png':
        source = os.path.join(basePath, item)
        newname = f'{tSA}_{prompts_mode}_{item}'
        destination = os.path.join('result', newname)
        shutil.copy2(source, destination)

shutil.rmtree(f'{tSA}')


