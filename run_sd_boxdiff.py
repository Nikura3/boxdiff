import os
import pathlib
import pprint
import time
from typing import List

import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils, vis_utils, logger
from utils.ptp_utils import AttentionStore
import torchvision.utils
import torchvision.transforms.functional as tf

import numpy as np
from utils.drawer import draw_rectangle, DashedImageDraw

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
        # If you cannot access the huggingface on your server, you can use the local prepared one.
        # stable_diffusion_version = "../../packages/huggingface/hub/stable-diffusion-v1-4"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version).to(device)

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
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
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
                    config=config,
                    height=512,
                    width=512)
    image = outputs.images[0]
    return image


#@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    # intialize logger
    l = logger.Logger(config.output_path)

    gen_images = []
    for seed in config.seeds:
        print(f"Current seed is : {seed}")

        # start stopwatch
        start = time.time()

        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)

        # end stopwatch
        end = time.time()
        # save to logger
        l.log_time_run(start, end)

        gen_images.append(image)
        image.save(str(config.output_path) +"/"+ str(seed) + ".jpg")

        #draw the bounding boxes
        image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),torch.Tensor(bbox[sample_to_generate]),labels=phrases[sample_to_generate],colors=['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black', 'gray', 'white'],width=4)
        tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(gen_images)
    # joined_image.save(config.output_path / f'{config.prompt}.png')
    joined_image.save(str(config.output_path) + "/" + config.prompt + ".png")


if __name__ == '__main__':
    height = 512
    width = 512
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    prompts = ["A small red brown and white dog catches a football in midair as a man and child look on .", #0
               "A small red brown and white dog catches a football in midair as a man and child look on .", #1
               "A black dog with his purple tongue sticking out running on the beach with a white dog wearing a red collar .", #2
               "a yellow chair and a blue cat", #3
               "two cups filled with steaming hot coffee sit side-by-side on a wooden table.", #4
               "a cup beside another cup are filled with steaming hot coffee on a wooden table.", #5
               "a red cat, a yellow banana, a blue dog and a orange horse.", #6
               "a orange horse, a blue dog, a yellow banana, a red cat.", #7
               "A tropical beach with crystal-clear water, a beach kiosk, and a hammock hanging between two palm trees.", #8
               "A young girl sitting on a picnic blanket under an oak tree, reading a colorful storybook.", #9
               "A brown horse with long ears is riding through a forest while a monkey with a hat is sitting on a branch.", #10
               "A white truck is parked on the beach with a surfboard strapped to its roof.", #11
               "An airplane is flying in the distance in a blue sky while a kite flies in the air controlled by a child."] #12

    bbox = [[[108, 242, 165, 457], [216, 158, 287, 351], [325, 98, 503, 508], [200, 175, 232, 250]],#0
            [[108, 242, 165, 457], [216, 158, 287, 351], [325, 98, 503, 508], [170, 175, 202, 250]],#1
            [[118, 258, 138, 284], [343, 196, 388, 267], [97, 147, 173, 376], [2, 31, 509, 508], [329, 157, 391, 316]],#2
            [[58,63,238,427],[297,218,464,417]],#3
            [[64,94,230,254],[254,137,356,258]],#4
            [[64,94,230,254],[254,137,356,258]],#5
            [[35,35,143,170],[344,406,510,508],[48,270,336,501],[172,56,474,382]],#6
            [[172,56,474,382],[48,270,336,501],[344,406,510,508],[35,35,143,170]],#7
            [[0,81,509,510],[11,45,224,298],[205,308,409,382],[126,210,209,459],[416,210,490,469]],#8
            [[214,292,312,350],[61,344,469,469],[35,18,486,320],[256,373,281,414]],#9
            [[53,138,466,328],[337,200,390,341],[39,80,153,347],[73,57,125,103]],#10
            [[68,209,402,459],[107,137,372,208]],#11
            [[63,47,120,90],[356,105,391,142],[418,272,452,321]]#12
            ]

    phrases = [["child", "A small red brown and white dog", "a man", "a football"],#0
               ["child", "A small red brown and white dog", "a man", "a football"],#1
               ["his purple tongue", "a red collar", "A black dog", "the beach", "a white dog"],#2
               ["chair","cat"],#3
               ["cup","cup,"],#4
               ["cup","cup"],#5
               ["cat","banana","dog","horse"],#6
               ["horse","dog","banana","cat"],
               ["beach","kiosk","hammock","tree","tree"],
               ["girl","blanket","tree","storybook"],
               ["horse","ears","monkey","hat"],
               ["truck","surfboard"],
               ["airplane","kite","child"]
               ]

    token_indices = [[18, 7, 16, 10],#0
                     [18, 7, 16, 10],#1
                     [7, 21, 3, 13, 17],#2
                     [3,7],#3
                     [2,2],#4
                     [2,5],#5
                     [3,7,11,15],#6
                     [15,11,7,3],
                     [3,12,16,21,21],
                     [3,8,12,18],
                     [3,6,14,17],
                     [3,12],
                     [2,14,22]]

    #Stable Diffusion 1.4
    model_name="BD_SD14"
    for sample_to_generate in range(0,13):
        output_path = "./results/"+model_name+"/"+ prompts[sample_to_generate] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",sample_to_generate)
        torch.cuda.empty_cache()
        main(RunConfig(
            prompt=prompts[sample_to_generate],
            gligen_phrases=phrases[sample_to_generate],
            seeds=seeds,
            token_indices=token_indices[sample_to_generate],
            bbox=bbox[sample_to_generate],
            output_path=output_path,
            sd_2_1=False
        ))

    #Stable Diffusion 2
    model_name="BD_SD2"
    for sample_to_generate in range(0,13):
        output_path = "./results/"+model_name+"/"+ prompts[sample_to_generate] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",sample_to_generate)
        torch.cuda.empty_cache()
        main(RunConfig(
            prompt=prompts[sample_to_generate],
            gligen_phrases=phrases[sample_to_generate],
            seeds=seeds,
            token_indices=token_indices[sample_to_generate],
            bbox=bbox[sample_to_generate],
            output_path=output_path,
            sd_2_1=True
        ))  