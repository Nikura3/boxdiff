import os
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

import torchvision.utils
import torchvision.transforms.functional as tf

import numpy as np
from utils.drawer import draw_rectangle, DashedImageDraw

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def make_tinyHRS():
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
            [[214,250,312,450],[61,344,469,469],[35,18,486,320],[256,373,281,414]],#9
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
               ["horse","dog","banana","cat"],#7
               ["beach","kiosk","hammock","tree","tree"],#8
               ["girl","blanket","tree","storybook"],#9
               ["horse","ears","monkey","hat"],#10
               ["truck","surfboard"],#11
               ["airplane","kite","child"]#12
               ]

    token_indices = [[18, 7, 16, 10],#0
                     [18, 7, 16, 10],#1
                     [7, 21, 3, 13, 17],#2
                     [3,7],#3
                     [2,2],#4
                     [2,5],#5
                     [3,7,11,15],#6
                     [15,11,7,3],#7
                     [3,12,16,21,21],#8
                     [3,8,12,18],#9
                     [3,6,14,17],#10
                     [3,12],#11
                     [2,14,22]#12
                     ]
    data_dict = {
    i: {
        "prompt": prompts[i],
        "bbox": bbox[i],
        "phrases": phrases[i],
        "token_indices": token_indices[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def sample():
    prompts = ["A bear wearing sunglasses and a green tie looks very proud" #0
               ]
    

    bbox = [[[67,87,366,512],[66,130,364,262],[178,327,266,512]]#0
            ]

    phrases = [["bear","sunglasses","green tie"]]

    token_indices = [[2,4,7],#0
                     ]
    data_dict = {
    i: {
        "prompt": prompts[i],
        "bbox": bbox[i],
        "phrases": phrases[i],
        "token_indices": token_indices[i]
    }
    for i in range(len(prompts))
    }
    return data_dict


def make_QBench():

    prompts = ["A bus", #0
               "A bus and a bench", #1
               "A bus next to a bench and a bird", #2
               "A bus next to a bench with a bird and a pizza", #3
               "A green bus", #4
               "A green bus and a red bench", #5
               "A green bus next to a red bench and a pink bird", #6
               "A green bus next to a red bench with a pink bird and a yellow pizza", #7
               "A bus on the left of a bench", #8
               "A bus on the left of a bench and a bird", #9
               "A bus and a pizza on the left of a bench and a bird", #10
               "A bus and a pizza on the left of a bench and below a bird", #11
               ]

    ids = []

    for i in range(len(prompts)):
        ids.append(str(i).zfill(3))
    

    bboxes = [[[2,121,251,460]],#0
            [[2,121,251,460], [274,345,503,496]],#1
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#2
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#3
            [[2,121,251,460]],#4
            [[2,121,251,460], [274,345,503,496]],#5
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#6
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#7
            [[2,121,251,460],[274,345,503,496]],#8
            [[2,121,251,460],[274,345,503,496],[344,32,500,187]],#9
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#10
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#11
            ]

    phrases = [["bus"],#0
               ["bus", "bench"],#1
               ["bus", "bench", "bird"],#2
               ["bus","bench","bird","pizza"],#3
               ["bus"],#4
               ["bus", "bench"],#5
               ["bus", "bench", "bird"],#6
               ["bus","bench","bird","pizza"],#7
               ["bus","bench"],#8
               ["bus","bench","bird"],#9
               ["bus","pizza","bench","bird"],#11
               ["bus","pizza","bench","bird"]#12
               ]

    token_indices = [[2],#0
                     [2,5],#1
                     [2, 6, 9],#2
                     [2,6,9,12],#3
                     [3],#4
                     [3,7],#5
                     [3,8,12],#6
                     [3,8,12,16],#7
                     [2,8],#8
                     [2,8,11],#9
                     [2,5,11,14],#10
                     [2,5,11,15],#11
                     ]
    data_dict = {
    i: {
        "id": ids[i],
        "prompt": prompts[i],
        "bboxes": bboxes[i],
        "phrases": phrases[i],
        "token_indices": token_indices[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable_diffusion_version = "masterful/gligen-1-4-generation-text-box"
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

#TODO best matching
def assign_indices_to_alter(stable, prompt: str,gligen_phrase:List[str] ) -> List[int]:
    print(stable.tokenizer(prompt)['input_ids'])
    input()
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
    for i in range(len(config.bboxes)):
        x1, y1, x2, y2 = config.bboxes[i]
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
                    bbox=config.bboxes,
                    height=512,
                    width=512,
                    config=config,
                    #negative_prompt='collage, split frame, frame, border, tiled, multiple images, diptych, triptych'
                    )
    image = outputs.images[0]
    return image


#@pyrallis.wrap()
def main(config: RunConfig):

    stable = load_model()
    token_indices = get_indices_to_alter(stable, config.prompt, config.gligen_phrases) if config.token_indices is None else config.token_indices
    
    #intialize logger
    l=logger.Logger(config.output_path)

    gen_images = []
    gen_bboxes_images=[]
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


        #image.save(prompt_output_path / f'{seed}.png')
        image.save(str(config.output_path) +"/"+ str(seed) + ".jpg")
        #list of tensors
        gen_images.append(tf.pil_to_tensor(image))

        
        #draw the bounding boxes
        image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),torch.Tensor(bench[sample_to_generate]['bboxes']),labels=bench[sample_to_generate]['phrases'],colors=['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black', 'gray', 'white'],width=4)
        #list of tensors
        gen_bboxes_images.append(image)
        tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(config.prompt)

    # save a grid of results across all seeds without bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + ".png")
    #joined_image = vis_utils.get_image_grid(gen_images)
    #joined_image.save(str(config.output_path) +"/"+ config.prompt + ".png")

    # save a grid of results across all seeds with bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")
    #joined_image = vis_utils.get_image_grid(gen_bboxes_images)
    #joined_image.save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")


if __name__ == '__main__':
    height = 512
    width = 512
    seeds = range(1,17)

    #bench=make_tinyHRS()
    bench=make_QBench()

    model_name="QBench-BD_G"
    for sample_to_generate in range(0,len(bench)):
        output_path = "./results/"+model_name+"/"+ bench[sample_to_generate]['id']+'_'+bench[sample_to_generate]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)
        print("Sample number ",sample_to_generate)
        torch.cuda.empty_cache()
        main(RunConfig(
            prompt_id=bench[sample_to_generate]['id'],
            prompt=bench[sample_to_generate]['prompt'],
            gligen_phrases=bench[sample_to_generate]['phrases'],
            seeds=seeds,
            token_indices=bench[sample_to_generate]['token_indices'],
            bboxes=bench[sample_to_generate]['bboxes'],
            output_path=output_path,
        )) 
