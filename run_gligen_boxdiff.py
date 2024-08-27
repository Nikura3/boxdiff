import pathlib
import pprint
import time
from typing import List

import prompt_toolkit.layout
#import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.gligen_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils, vis_utils, logger
from utils.ptp_utils import AttentionStore

import numpy as np
from utils.drawer import draw_rectangle, DashedImageDraw

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #stable_diffusion_version = "gligen/diffusers-generation-text-box"
    stable_diffusion_version = "masterful/gligen-1-4-generation-text-box"
    # If you cannot access the huggingface on your server, you can use the local prepared one.
    # stable_diffusion_version = "../../packages/diffusers/gligen_ckpts/diffusers-generation-text-box"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version, use_safetensors=False ,local_files_only=True).to(device)

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: BoxDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:

    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    gligen_boxes = []
    for i in range(len(config.bbox)):
        x1, y1, x2, y2 = config.bbox[i]
        gligen_boxes.append([x1/512, y1/512, x2/512, y2/512])

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    gligen_phrases=config.gligen_phrases,
                    gligen_boxes=gligen_boxes,
                    gligen_scheduled_sampling_beta=0.3,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    bbox=config.bbox,
                    height=512,
                    width=512,
                    config=config)
    image = outputs.images[0]
    return image


#@pyrallis.wrap()
def main(config: RunConfig):

    stable = load_model()
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices
    #intialize logger
    l=logger.Logger(config.output_path)

    if len(config.bbox[0]) == 0:
        config.bbox = draw_rectangle()

    images = []
    for seed in config.seeds:
        print(f"Current seed is : {seed}")

        #start stopwatch
        start=time.time()

        if torch.cuda.is_available():
            g = torch.Generator('cuda').manual_seed(seed)
        else:
            g = torch.Generator('cpu').manual_seed(seed)
        controller = AttentionStore()
        controller.num_uncond_att_layers = -16
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)
        #end stopwatch
        end = time.time()
        #save to logger
        l.log_time_run(start,end)

        prompt_output_path = config.output_path / config.prompt[:100]
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

        canvas = Image.fromarray(np.zeros((image.size[0], image.size[0], 3), dtype=np.uint8) + 220)
        draw = DashedImageDraw(canvas)

        for i in range(len(config.bbox)):
            x1, y1, x2, y2 = config.bbox[i]
            draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[i], width=5)
        canvas.save(prompt_output_path / f'{seed}_canvas.png')

    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(config.prompt)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    prompt_collection=[
        RunConfig(prompt="A rabbit wearing sunglasses looks very proud",
                       gligen_phrases= ["a rabbit", "sunglasses"],
                       seeds=[1,2],
                       P=0.2,
                       L=1,
                       refine=False,
                       token_indices=[2,4],
                       #bbox=[[34, 44, 183, 256],[33, 65, 182, 131]], #256
                       bbox= [[67, 87, 366, 512], [66, 130, 364, 262]], #512
                       output_path=pathlib.Path("BoxDiff_GLIGEN/")),
        RunConfig(prompt="A small red brown and white dog catches a football in midair as a man and child look on",
                       gligen_phrases= ["child","A small red brown and white dog","a man","a football"],
                       seeds=[1, 2, 3, 4,5,6,7,8,9],
                       P=0.2,
                       L=1,
                       refine=False,
                       token_indices=[1,2,3,4],
                       #bbox=[[34, 44, 183, 256],[33, 65, 182, 131]], #256
                       bbox= [[108,242,165,457],#child
                              [216,158,287,351],#a small red brown and white dog
                              [325,98,503,508],#a man
                              [200,175,232,250]# a football
                              ],
                       output_path=pathlib.Path("BoxDiff_GLIGEN/"))
    ]
    main(prompt_collection[0])
