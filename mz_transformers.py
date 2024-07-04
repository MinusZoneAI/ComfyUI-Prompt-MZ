import json
import os
import shutil
import subprocess
import traceback


def transformers_captioner(args_dict, myfunc):
    from . import mz_prompt_utils
    import PIL.Image as Image
    captioner_config = args_dict.get("captioner_config", {})
    directory = captioner_config.get("directory", None)
    force_update = captioner_config.get("force_update", False)
    caption_suffix = captioner_config.get("caption_suffix", "")
    retry_keyword = captioner_config.get("retry_keyword", "")
    batch_size = captioner_config.get("batch_size", 1)
    retry_keywords = retry_keyword.split(",")

    retry_keywords = [k.strip() for k in retry_keywords]
    retry_keywords = [k for k in retry_keywords if k != ""]

    pre_images = []
    # print("directory:", directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                base_file_path = os.path.splitext(image_path)[0]
                caption_file = os.path.join(
                    root, base_file_path + caption_suffix)
                if os.path.exists(caption_file) and force_update is False:
                    continue

                pre_images.append({
                    "image_path": image_path,
                    "caption_path": caption_file
                })

    result = []

    # print(f"Total images: {len(pre_images)} : {json.dumps(pre_images, indent=4)}")
    print(f"Total images: {len(pre_images)}")

    pb = mz_prompt_utils.Utils.progress_bar(len(pre_images))
    images_batch = []
    for i in range(len(pre_images)):
        try:
            pre_image = pre_images[i]
            image_path = pre_image["image_path"]
            caption_file = pre_image["caption_path"]

            onec_args_dict = args_dict.copy()
            del onec_args_dict["captioner_config"]

            pil_image = Image.open(image_path)
            if len(images_batch) < batch_size:
                images_batch.append({
                    "image_path": image_path,
                    "pil_image": pil_image
                })
                if i < len(pre_images) - 1:
                    continue

            if i < len(pre_images) - 1:
                onec_args_dict["keep_device"] = True

            pil_images = []
            for j in range(len(images_batch)):
                pil_images.append(images_batch[j]["pil_image"])

            # onec_args_dict["image"] = mz_prompt_utils.Utils.pil2tensor(
            #     pil_image)

            thumbnail = Image.new(
                "RGB", (images_batch[0]["pil_image"].width * batch_size, images_batch[0]["pil_image"].height))

            for j in range(len(images_batch)):
                pil_image = images_batch[j]["pil_image"]
                thumbnail.paste(pil_image, (j * pil_image.width, 0))

            pb.update(
                i,
                len(pre_images),
                # 转RGB
                thumbnail,
            )
            onec_args_dict["images"] = pil_images
            onec_args_dict["captioner_mode"] = True

            responses = myfunc(onec_args_dict)
            # print(f"responses: {responses}")
            for j in range(len(images_batch)):
                item = images_batch[j]
                image_path = item["image_path"]
                caption_file = os.path.join(
                    os.path.dirname(image_path), os.path.splitext(image_path)[0] + caption_suffix)
                response = responses[j]
                response = response.strip()

                print(f"==={image_path}===")
                print(image_path)
                print(response)
                print("")
                print("")

                if response != "":
                    with open(caption_file, "w") as f:
                        prompt_fixed_beginning = captioner_config.get(
                            "prompt_fixed_beginning", "")
                        f.write(prompt_fixed_beginning + response)

                result.append(response)

            images_batch = []
        except Exception as e:
            print(
                f"For image {image_path}, error: {e} , stack: {traceback.format_exc()}")
    return result


def florence2_node_encode(args_dict):
    args_dict = args_dict.copy()
    captioner_config = args_dict.get("captioner_config", None)
    if captioner_config is not None:
        transformers_captioner(args_dict, florence2_node_encode)
        # raise Exception(
        #     "图片批量反推任务已完成 ; Image batch reverse push task completed")
        return {"ui": {"string": ["图片批量反推任务已完成 ; Image batch reverse push task completed",]}, "result": ("", None)}

    import torch
    import folder_paths
    from . import mz_prompt_utils
    from .mz_prompt_utils import Utils

    florence2_large_files_map = {
        "Florence-2-large": [
            {
                "file_path": "pytorch_model.bin",
                "url": "https://www.modelscope.cn/api/v1/models/AI-ModelScope/Florence-2-large/repo?Revision=master&FilePath=pytorch_model.bin"
            }
        ],
        "Florence-2-large-ft": [
            {
                "file_path": "pytorch_model.bin",
                "url": "https://www.modelscope.cn/api/v1/models/AI-ModelScope/Florence-2-large-ft/repo?Revision=master&FilePath=pytorch_model.bin"
            }
        ],
    }

    llm_path = os.path.join(
        folder_paths.models_dir,
        "LLM",
    )
    os.makedirs(llm_path, exist_ok=True)

    model_name = args_dict.get("model_name", "Florence-2-large")

    model_path = os.path.join(llm_path, model_name)

    if not os.path.exists(model_path):
        # GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/AI-ModelScope/Florence-2-large.git
        original_env = os.environ.get("GIT_LFS_SKIP_SMUDGE")
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            ["git", "clone", "https://www.modelscope.cn/AI-ModelScope/Florence-2-large.git", model_path])
        if original_env is not None:
            os.environ["GIT_LFS_SKIP_SMUDGE"] = original_env

    florence2_large_files = florence2_large_files_map.get(model_name, [])
    for file_info in florence2_large_files:
        file_path = os.path.join(model_path, file_info["file_path"])
        # 判断文件大小小于1M
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024 * 1024:
            Utils.download_file(file_info["url"], file_path)

    # with open(os.path.join(os.path.dirname(__file__), "hook", "modeling_florence2.py"), "r") as f:
    #     code = f.read()
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__),
                     "hook", "modeling_florence2.py"),
        os.path.join(model_path, "modeling_florence2.py")
    )

    from transformers import AutoProcessor, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Utils.cache_get(f"florence_model_and_opt_")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model.to(device).eval()
        Utils.cache_set(f"florence_model_and_opt_", model)

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    captioner_mode = args_dict.get("captioner_mode", False)
    if captioner_mode:
        pil_images = args_dict.get("images", None)
        _pil_images = []
        for pil_image in pil_images:
            resolution = args_dict.get("resolution", 512)
            pil_image = Utils.resize_max(
                pil_image, resolution, resolution).convert("RGB")
            _pil_images.append(pil_image)
        pil_images = _pil_images
    else:
        tensor_image = args_dict.get("image", None)
        pil_image = Utils.tensor2pil(tensor_image)
        resolution = args_dict.get("resolution", 512)
        pil_image = Utils.resize_max(
            pil_image, resolution, resolution).convert("RGB")
        pil_images = [pil_image]

    prompt = "<MORE_DETAILED_CAPTION>"
    prompts = [prompt for _ in pil_images]
    inputs = processor(text=prompts, images=pil_images, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    generated_texts = processor.batch_decode(
        generated_ids, skip_special_tokens=True)

    pil_image = pil_images[0]
    parsed_answers = []
    for i in range(len(generated_texts)):
        generated_text = generated_texts[i]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(pil_image.width, pil_image.height))
        parsed_answers.append(parsed_answer)

    response = []
    for i in range(len(parsed_answers)):
        response.append(parsed_answers[i].get(prompt))

    keep_device = args_dict.get("keep_device", False)
    if not keep_device:
        model.cpu()
        del model
        torch.cuda.empty_cache()
        Utils.cache_set(f"florence_model_and_opt_", None)

    if captioner_mode:
        return response
    else:
        response = response[0]

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [mz_prompt_utils.Utils.to_debug_prompt(response),]}, "result": (response, conditionings)}


def paligemma_node_encode(args_dict):
    args_dict = args_dict.copy()
    captioner_config = args_dict.get("captioner_config", None)
    if captioner_config is not None:
        transformers_captioner(args_dict, paligemma_node_encode)
        # raise Exception(
        #     "图片批量反推任务已完成 ; Image batch reverse push task completed")
        return {"ui": {"string": ["图片批量反推任务已完成 ; Image batch reverse push task completed",]}, "result": ("", None)}

    import torch
    import folder_paths
    from . import mz_prompt_utils
    from .mz_prompt_utils import Utils

    paligemma_files_map = {
        "common": [

            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fadded_tokens.json",
                "file_path": "added_tokens.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fconfig.json",
                "file_path": "config.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fgeneration_config.json",
                "file_path": "generation_config.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fpreprocessor_config.json",
                "file_path": "preprocessor_config.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fspecial_tokens_map.json",
                "file_path": "special_tokens_map.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Ftokenizer.json",
                "file_path": "tokenizer.json"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Ftokenizer.model",
                "file_path": "tokenizer.model"
            },
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Ftokenizer_config.json",
                "file_path": "tokenizer_config.json"
            },
        ],
        "paligemma-sd3-long-captioner": [
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-merge%2Fmodel.safetensors",
                "file_path": "model.safetensors"
            },
        ],
        "paligemma-sd3-long-captioner-v2": [
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sd3-long-captioner-v2-merge%2Fmodel.safetensors",
                "file_path": "model.safetensors"
            },
        ],
        "paligemma-sdxl-long-captioner": [
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/sd-models/repo?Revision=master&FilePath=sdxl-long-captioner-merge%2Fmodel.safetensors",
                "file_path": "model.safetensors"
            },
        ],
    }

    llm_path = os.path.join(
        folder_paths.models_dir,
        "LLM",
    )
    os.makedirs(llm_path, exist_ok=True)

    model_name = args_dict.get("model_name")

    model_path = os.path.join(llm_path, model_name)

    common_files = paligemma_files_map.get("common", [])
    for file_info in common_files:
        file_path = os.path.join(model_path, file_info["file_path"])
        if not os.path.exists(file_path):
            Utils.download_file(file_info["url"], file_path)

    paligemma_files = paligemma_files_map.get(model_name, [])
    for file_info in paligemma_files:
        file_path = os.path.join(model_path, file_info["file_path"])

        if not os.path.exists(file_path):
            Utils.download_file(file_info["url"], file_path)

    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Utils.cache_get(f"paligemma_model_and_opt_")
    if model is None:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        model.to(device).eval()
        Utils.cache_set(f"paligemma_model_and_opt_", model)

    processor = PaliGemmaProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    tensor_image = args_dict.get("image")
    pil_image = Utils.tensor2pil(tensor_image)
    resolution = args_dict.get("resolution", 512)
    pil_image = Utils.resize_max(
        pil_image, resolution, resolution).convert("RGB")

    # prefix
    prompt = "caption en"
    model_inputs = processor(
        text=prompt, images=pil_image, return_tensors="pt").to('cuda')
    input_len = model_inputs["input_ids"].shape[-1]

    def modify_caption(caption: str) -> str:
        """
        Removes specific prefixes from captions.
        Args:
            caption (str): A string containing a caption.
        Returns:
            str: The caption with the prefix removed if it was present.
        """
        # Define the prefixes to remove
        import re
        prefix_substrings = [
            ('captured from ', ''),
            ('captured at ', '')
        ]

        # Create a regex pattern to match any of the prefixes
        pattern = '|'.join([re.escape(opening)
                           for opening, _ in prefix_substrings])
        replacers = {opening: replacer for opening,
                     replacer in prefix_substrings}

        # Function to replace matched prefix with its corresponding replacement
        def replace_fn(match):
            return replacers[match.group(0)]

        # Apply the regex to the caption
        return re.sub(pattern, replace_fn, caption, count=1, flags=re.IGNORECASE)

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs, max_new_tokens=256, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

        modified_caption = modify_caption(decoded)
        # print(modified_caption)

    response = modified_caption

    keep_device = args_dict.get("keep_device", False)
    if not keep_device:
        model.cpu()
        del model
        torch.cuda.empty_cache()
        Utils.cache_set(f"paligemma_model_and_opt_", None)

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [mz_prompt_utils.Utils.to_debug_prompt(response),]}, "result": (response, conditionings)}
