
import importlib
import json
import os
from . import mz_prompt_utils
from . import mz_llama_cpp
from . import mz_prompts


def get_schema_base_type(t):
    return {
        "type": t,
    }


def get_schema_obj(keys_type={}, required=[]):
    item = {}
    for key, value in keys_type.items():
        if type(value) == str:
            value = get_schema_base_type(value)
        item[key] = value
    return {
        "type": "object",
        "properties": item,
        "required": required
    }


def get_schema_array(item_type="string"):
    if type(item_type) == str:
        item_type = get_schema_base_type(item_type)
    return {
        "type": "array",
        "items": item_type,
    }


high_quality_prompt = "((high quality:1.4), (best quality:1.4), (masterpiece:1.4), (8K resolution), (2k wallpaper))"
style_presets_prompt = {
    "none": "",
    "high_quality": high_quality_prompt,
    "photography": f"{high_quality_prompt}, (RAW photo, best quality), (realistic, photo-realistic:1.2), (bokeh, cinematic shot, dynamic composition, incredibly detailed, sharpen, details, intricate detail, professional lighting, film lighting, 35mm, anamorphic, lightroom, cinematography, bokeh, lens flare, film grain, HDR10, 8K)",
    "illustration": f"{high_quality_prompt}, ((detailed matte painting, intricate detail, splash screen, complementary colors), (detailed),(intricate details),illustration,an extremely delicate and beautiful,ultra-detailed,highres,extremely detailed)",
}


def get_style_presets():
    return [
        "none",
        "high_quality",
        "photography",
        "illustration",
    ]


def llama_cpp_node_encode(args_dict):
    importlib.reload(mz_prompts)
    importlib.reload(mz_llama_cpp)

    model_config = args_dict.get("llama_cpp_model", {})
    mz_prompt_utils.Utils.print_log(f"model_config: {model_config}")

    select_model_type = model_config.get("type", "ManualSelect")
    if select_model_type == "ManualSelect":
        model_file = model_config.get("model_path", "auto")
        if model_file == "auto":
            model_file = mz_prompt_utils.Utils.get_auto_model_fullpath(
                "Meta-Llama-3-8B-Instruct-Q4_K_M")
    elif select_model_type == "DownloaderSelect":
        model_name = model_config.get("model_name", "")
        model_file = mz_prompt_utils.Utils.get_auto_model_fullpath(
            model_name)
    else:
        raise Exception("Unknown select_model_type")

    mz_prompt_utils.Utils.print_log(f"model_file: {model_file}")

    text = args_dict.get("text", "")
    style_presets = args_dict.get("style_presets", "")
    options = args_dict.get("llama_cpp_options", {})
    keep_device = args_dict.get("keep_device", False)
    seed = args_dict.get("seed", -1)
    options["seed"] = seed

    customize_instruct = args_dict.get("customize_instruct", None)
    mz_prompt_utils.Utils.print_log(
        f"customize_instruct: {customize_instruct}")
    try:
        schema = None
        if customize_instruct is None:
            schema = get_schema_obj(
                keys_type={
                    "description": get_schema_base_type("string"),
                    "long_prompt": get_schema_base_type("string"),
                    "main_color_word": get_schema_base_type("string"),
                    "camera_angle_word": get_schema_base_type("string"),
                    "style_words": get_schema_array("string"),
                    "subject_words": get_schema_array("string"),
                    "light_words": get_schema_array("string"),
                    "environment_words": get_schema_array("string"),
                },
                required=[
                    "description",
                    "long_prompt",
                    "main_color_word",
                    "camera_angle_word",
                    "style_words",
                    "subject_words",
                    "light_words",
                    "environment_words",
                ]
            )

            question = f"IDEA: {style_presets},{text}"
            if style_presets == "none":
                question = f"IDEA: {text}"

            system_prompt = mz_prompts.Beautify_Prompt + mz_prompts.Long_prompt + "\n"

        else:

            system_prompt = customize_instruct.get("system", "")
            question = customize_instruct.get("instruct", "%text%")

            system_prompt = system_prompt.replace("%text%", text)
            question = question.replace("%text%", text)

            mz_prompt_utils.Utils.print_log(f"system_prompt: {system_prompt}")
            mz_prompt_utils.Utils.print_log(f"question: {question}")

        if schema is not None:
            response_json = mz_llama_cpp.llama_cpp_simple_interrogator_to_json(
                model_file=model_file,
                system=system_prompt,
                question=question,
                schema=schema,
                options=options,
            )
            mz_prompt_utils.Utils.print_log(f"response_json: {response_json}")

            response = json.loads(response_json)
            full_responses = []

            if response["description"] != "":
                full_responses.append(f"({response['description']})")
            if response["long_prompt"] != "":
                full_responses.append(f"({response['long_prompt']})")
            if response["main_color_word"] != "":
                full_responses.append(f"({response['main_color_word']})")
            if response["camera_angle_word"] != "":
                full_responses.append(f"({response['camera_angle_word']})")

            response["style_words"] = [
                x for x in response["style_words"] if x != ""]
            if len(response["style_words"]) > 0:
                full_responses.append(
                    f"({', '.join(response['style_words'])})")

            response["subject_words"] = [
                x for x in response["subject_words"] if x != ""]
            if len(response["subject_words"]) > 0:
                full_responses.append(
                    f"({', '.join(response['subject_words'])})")

            response["light_words"] = [
                x for x in response["light_words"] if x != ""]
            if len(response["light_words"]) > 0:
                full_responses.append(
                    f"({', '.join(response['light_words'])})")

            response["environment_words"] = [
                x for x in response["environment_words"] if x != ""]
            if len(response["environment_words"]) > 0:
                full_responses.append(
                    f"({', '.join(response['environment_words'])})")

            full_response = ", ".join(full_responses)
        else:
            full_response = mz_llama_cpp.llama_cpp_simple_interrogator(
                model_file=model_file,
                system=system_prompt,
                question=question,
                options=options,
            )

            start_str = customize_instruct.get("start_str", "")
            if start_str != "" and full_response.find(start_str) != -1:
                full_response_list = full_response.split(start_str)
                # 删除第一个元素
                full_response_list.pop(0)
                full_response = start_str.join(full_response_list)

            end_str = customize_instruct.get("end_str", "")
            if end_str != "" and full_response.find(end_str) != -1:
                full_response_list = full_response.split(end_str)
                # 删除最后一个元素
                full_response_list.pop()
                full_response = end_str.join(full_response_list)

        if keep_device is False:
            mz_llama_cpp.freed_gpu_memory(model_file=model_file)

        # 去除换行
        while full_response.find("\n") != -1:
            full_response = full_response.replace("\n", " ")

        # 句号换成逗号
        while full_response.find(".") != -1:
            full_response = full_response.replace(".", ",")

        # 去除多余逗号
        while full_response.find(",,") != -1:
            full_response = full_response.replace(",,", ",")
        while full_response.find(", ,") != -1:
            full_response = full_response.replace(", ,", ",")

        full_response = mz_prompt_utils.Utils.prompt_zh_to_en(full_response)

        style_presets_prompt_text = style_presets_prompt.get(style_presets, "")

        if style_presets_prompt_text != "":
            full_response = f"{style_presets_prompt_text}, {full_response}"

    except Exception as e:
        mz_llama_cpp.freed_gpu_memory(model_file=model_file)
        raise e

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = mz_prompt_utils.Utils.a1111_clip_text_encode(
            clip, full_response, )

    return {"ui": {"string": [full_response,]}, "result": (full_response, conditionings)}


def image_interrogator_captioner(args_dict):
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
        onec_args_dict["image"] = pil_image

        if i < len(pre_images) - 1:
            onec_args_dict["keep_device"] = True

        pb.update(
            i,
            len(pre_images),
            pil_image.copy(),
        )

        response = image_interrogator_node_encode(onec_args_dict)
        response = response.get("result", ())[0]
        response = response.strip()
        is_retry = response == ""
        for k in retry_keywords:
            if response.find(k) != -1:
                print(f"存在需要重试的关键词 ; Retry keyword found: {k}")
                is_retry = True
                break

        mz_prompt_utils.Utils.print_log(
            "\n\nonec_args_dict: ", onec_args_dict)
        if is_retry:
            for retry_n in range(5):
                print(f"Retry {retry_n+1}...")
                onec_args_dict["seed"] = onec_args_dict["seed"] + 1
                response = image_interrogator_node_encode(onec_args_dict)
                response = response.get("result", ())[0]
                response = response.strip()
                is_retry = response == ""
                for k in retry_keywords:
                    if response.find(k) != -1:
                        print(f"存在需要重试的关键词 ; Retry keyword found: {k}")
                        is_retry = True
                        break

                if is_retry is False:
                    break
            if is_retry:
                print(f"重试失败,图片被跳过 ; Retry failed")
                response = ""

        if response != "":
            with open(caption_file, "w") as f:
                f.write(response)

        result.append(response)

        # mz_prompt_webserver.show_toast_success(
        #     f"提示词保存成功(prompt saved successfully): {caption_file}",
        #     1000,
        # )

    return result



image_interrogator_auto_mmproj_model = {
    
}

def image_interrogator_node_encode(args_dict):
    image_interrogator_model = args_dict.get("image_interrogator_model", {})

    llama_cpp_model = image_interrogator_model.get("llama_cpp_model", "auto")
    mmproj_model = image_interrogator_model.get("mmproj_model", "auto")

    captioner_config = args_dict.get("captioner_config", None)
    if captioner_config is not None:
        image_interrogator_captioner(args_dict)
        raise Exception(
            "图片批量反推任务已完成 ; Image batch reverse push task completed")

    image = args_dict.get("image", None)
    resolution = args_dict.get("resolution", 512)
    keep_device = args_dict.get("keep_device", False)
    seed = args_dict.get("seed", -1)
    options = args_dict.get("llama_cpp_options", {})
    options["seed"] = seed

    image = mz_prompt_utils.Utils.resize_max(image, resolution, resolution)

    customize_instruct = args_dict.get("customize_instruct", None)
    if customize_instruct is None:
        system_text = mz_prompts.GPT4VImageCaptioner_System
        question_text = mz_prompts.GPT4VImageCaptioner_Prompt
    else:

        system_prompt = customize_instruct.get("system", "")
        question = customize_instruct.get("instruct", "%text%")
        system_prompt = system_prompt.replace("%text%", "")
        question = question.replace("%text%", "")

    response = mz_llama_cpp.llava_cpp_simple_interrogator(
        model_file=llama_cpp_model,
        mmproj_file=mmproj_model,
        image=image,
        options=options,
        system=system_text,
        question=question_text,
    )

    if keep_device is False:
        mz_llama_cpp.freed_gpu_memory(model_file=llama_cpp_model)

    # return response

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = mz_prompt_utils.Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [response,]}, "result": (response, conditionings)}
