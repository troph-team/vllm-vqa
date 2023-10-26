from vllm import MLLM, SamplingParams
from PIL import Image

from pathlib import Path

import numpy as np
import glob
import os
import time
import argparse
import json

def batched(l: list, size: int) :
    for i in range(0, len(l), size) :
        yield l[i: i + size]

def main(src: str, user_prompt: str, model: str, n_label_per_image: int = 5, batch_size: int = 16) :
    model_type = ''
    model_config_json = None
    sampling_params = SamplingParams(n = n_label_per_image, temperature = 1.0, top_k = 5, top_p = 1, max_tokens = 512)
    if 'llava' in model.lower() :
        model_type = 'llava'
        with open(os.path.join(model, 'config.json'), 'r', encoding = 'utf-8') as fp :
            model_config_json = json.load(fp)
        model_config_json['model_type'] = 'llama'
        with open(os.path.join(model, 'config.json'), 'w', encoding = 'utf-8') as fp :
            json.dump(model_config_json, fp, indent = 2)
        gpu_memory_utilization = 0.9 # 90% of GPU
    elif 'qwen' in model.lower() :
        model_type = 'qwenvl'
        sampling_params.stop_token_ids.append(151645) # add <|im_end|> token
        gpu_memory_utilization = 0.95 # 95% of GPU
    else :
        assert False, 'Unknown model type'
    llm = MLLM(model = model, gpu_memory_utilization = gpu_memory_utilization, trust_remote_code = True)
    all_image_files = []
    for ext in ['.jpg', '.png', '.jpeg', '.webp', '.bmp'] :
        all_image_files.extend(glob.glob(os.path.join(src, '**/*' + ext)))
        all_image_files.extend(glob.glob(os.path.join(src, '*' + ext)))
    for batch in batched(all_image_files, batch_size) :
        images = [Image.open(filename).convert('RGB') for filename in batch]
        caption_files = []
        for filename in batch :
            p = Path(filename)
            [base, _] = os.path.split(filename)
            dst = os.path.join(base, f'{p.stem}.{model_type}.jsonl')
            caption_files.append(dst)
        if model_type == 'llava' :
            prompts = [f"A chat between a curious human and an artificial intelligence assistant. USER: <image>\n{user_prompt} ASSISTANT:" for i in range(len(batch))]
        elif model_type == 'qwenvl' :
            prompts = [f"Picture 1:<img>path</img>\n{user_prompt}" for i in range(len(batch))]
        start_ts = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, images = images)
        end_ts = time.perf_counter()
        print(f'Labelled {len(batch)} images at {len(prompts) / (end_ts-start_ts):0.2f} images/s')
        for capfile, outs in zip(caption_files, outputs) :
            with open(capfile, 'w', encoding = 'utf-8') as fp :
                for out in outs.outputs :
                    caption = out.text
                    if model_type == 'qwenvl' :
                        caption = caption.removesuffix('<|im_end|>')
                    logprob = out.cumulative_logprob
                    fp.write(json.dumps({'caption': caption, 'logprob': logprob}) + '\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'VQA labelling')
    parser.add_argument('--input', type=str, default = "/home/ubuntu/data/images")
    parser.add_argument('--model', type=str, default = "/home/ubuntu/data/models/llava-v1.5-7b")
    parser.add_argument('--prompt', type=str, default = "describe this image and its style in a very detailed manner.")
    parser.add_argument('--n', type=int, default = 1, help = 'How many captions to generate for an image')
    parser.add_argument('--bs', type=int, default = 256, help = 'Inference batch size')
    args = parser.parse_args()
    main(args.input, args.prompt, args.model, args.n, args.bs)
