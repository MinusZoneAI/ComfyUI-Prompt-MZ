import os
import shutil
import subprocess


def florence2_captioner(args_dict):
    from . import mz_prompt_utils
    import PIL.Image as Image
    captioner_config = args_dict.get("captioner_config", {})
    directory = captioner_config.get("directory", None)
    force_update = captioner_config.get("force_update", False)
    caption_suffix = captioner_config.get("caption_suffix", "")
    retry_keyword = captioner_config.get("retry_keyword", "")
    retry_keywords = retry_keyword.split(",")

    retry_keywords = [k.strip() for k in retry_keywords]
    retry_keywords = [k for k in retry_keywords if k != ""]

    pre_images = []
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

    pb = mz_prompt_utils.Utils.progress_bar(len(pre_images))
    for i in range(len(pre_images)):
        pre_image = pre_images[i]
        image_path = pre_image["image_path"]
        caption_file = pre_image["caption_path"]

        onec_args_dict = args_dict.copy()
        del onec_args_dict["captioner_config"]

        pil_image = Image.open(image_path)
        onec_args_dict["image"] = mz_prompt_utils.Utils.pil2tensor(pil_image)

        if i < len(pre_images) - 1:
            onec_args_dict["keep_device"] = True

        thumbnail = Image.new("RGB", (pil_image.width, pil_image.height))
        thumbnail.paste(pil_image)

        pb.update(
            i,
            len(pre_images),
            # 转RGB
            thumbnail,
        )

        response = florence2_node_encode(onec_args_dict)
        response = response.get("result", ())[0]
        response = response.strip()

        if response != "":
            with open(caption_file, "w") as f:
                f.write(response)

        result.append(response)
    return result


def florence2_node_encode(args_dict):
    args_dict = args_dict.copy()
    captioner_config = args_dict.get("captioner_config", None)
    if captioner_config is not None:
        florence2_captioner(args_dict)
        # raise Exception(
        #     "图片批量反推任务已完成 ; Image batch reverse push task completed")
        return {"ui": {"string": ["图片批量反推任务已完成 ; Image batch reverse push task completed",]}, "result": ("", None)}

    import torch
    import folder_paths
    from . import mz_prompt_utils
    from .mz_prompt_utils import Utils

    florence2_large_files = [
        {
            "file_path": "pytorch_model.bin",
            "url": "https://www.modelscope.cn/api/v1/models/AI-ModelScope/Florence-2-large/repo?Revision=master&FilePath=pytorch_model.bin"
        }
    ]

    # model_name = args_dict.get("model_name")
    # 判断是否是绝对路径
    # if os.path.isabs(model_name):
    #     model_path = model_name
    # else:
    llm_path = os.path.join(
        folder_paths.models_dir,
        "LLM",
    )
    os.makedirs(llm_path, exist_ok=True)
    model_path = os.path.join(llm_path, "Florence-2-large")

    if not os.path.exists(model_path):
        # GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/AI-ModelScope/Florence-2-large.git
        original_env = os.environ.get("GIT_LFS_SKIP_SMUDGE")
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            ["git", "clone", "https://www.modelscope.cn/AI-ModelScope/Florence-2-large.git", model_path])
        if original_env is not None:
            os.environ["GIT_LFS_SKIP_SMUDGE"] = original_env

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
        model.to(device)
        Utils.cache_set(f"florence_model_and_opt_", model)

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    tensor_image = args_dict.get("image")
    pil_image = Utils.tensor2pil(tensor_image)
    resolution = args_dict.get("resolution", 512)
    pil_image = Utils.resize_max(
        pil_image, resolution, resolution).convert("RGB")

    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(pil_image.width, pil_image.height))
    # print(f"parsed_answer: {parsed_answer}")
    response = parsed_answer.get(prompt)

    keep_device = args_dict.get("keep_device", False)
    if not keep_device:
        model.cpu()
        del model
        torch.cuda.empty_cache()
        Utils.cache_set(f"florence_model_and_opt_", None)

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [mz_prompt_utils.Utils.to_debug_prompt(response),]}, "result": (response, conditionings)}
