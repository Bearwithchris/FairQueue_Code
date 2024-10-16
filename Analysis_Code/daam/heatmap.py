from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple, Set

from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import os
import spacy.tokens
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import warnings

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
        else:
            heat_map = heat_map.clamp_(min=0, max=1)
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)

        # visualize the cross-attention map
        settings = torch.load('settings.pt')
        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        if word in ['S0','S1','S2'] and (settings['print_mode'] == 'HP_2_ITI, Stage 1 (HP)' or settings['print_mode'] == 'ITI_2_HP, Stage 2 (HP)' or settings['mode'] == 'HP'):
            word = settings['vocabulary'][word]
        plt_.imshow(im)
        plt_.axis('off')
        plt_.tight_layout(pad=0)
        plt.savefig('heatmap_temp.png')
        img = Image.open('heatmap_temp.png')
        os.remove('./heatmap_temp.png')
        width, height = img.size
        left = (width - 480) // 2
        top = (height - 480) // 2
        right = left + 480
        bottom = top + 480
        heatmap = img.crop((left, top, right, bottom))
        heatmap_resized = heatmap.resize((512, 512), Image.LANCZOS)
        title_image = Image.new('RGB', (512, 100), (255, 255, 255))
        font = ImageFont.truetype("Arial.ttf", 72)
        draw = ImageDraw.Draw(title_image)
        text = word
        text_width, text_height = draw.textsize(text, font=font)
        text_x = (title_image.width - text_width) // 2
        text_y = (title_image.height - text_height) // 2
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

        final_image = Image.new('RGB', (512, 612), (255, 255, 255))
        final_image.paste(title_image, (0, 0))
        final_image.paste(heatmap_resized, (0, 100))
        final_image.save(out_file)


class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None

        plot_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=None
        )

    def plot_overlay_hist_values(self, image, **expand_kwargs):
        return self.expand_as(image, **expand_kwargs)

    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        if word == 'A':
            word_idx = 0
        elif word == 'a':
            word_idx = 3
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)

RawHeatMapKey = Tuple[int, int, int, int]


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, inference_step: int, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (inference_step, factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def inference_step(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def factors(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[3] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
