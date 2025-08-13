
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
    # importlib.reload(mz_prompt_utils)

    model_config = args_dict.get("llama_cpp_model", {})
    mz_prompt_utils.Utils.print_log(f"model_config: {model_config}")

    chat_format = model_config.get("chat_format", None)

    select_model_type = model_config.get("type", "ManualSelect")
    if select_model_type == "ManualSelect":
        model_file = model_config.get("model_path", "auto")
        if model_file == "auto":
            model_file = mz_prompt_utils.Utils.get_auto_model_fullpath(
                "Meta-Llama-3-8B-Instruct.Q4_K_M")

            if "llama-3" in mz_llama_cpp.get_llama_cpp_chat_handlers():
                chat_format = "llama-3"

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
    translate_option = args_dict.get("translate", False)
    options["seed"] = seed
    options["chat_format"] = chat_format

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
            response_text = mz_llama_cpp.llama_cpp_simple_interrogator_to_json(
                model_file=model_file,
                system=system_prompt,
                question=question,
                schema=schema,
                options=options,
            )
            try:
                response_json = json.loads(response_text)
            except Exception as e:
                from . import half_json
                print("json.loads failed, try fix response_text: ", response_text)
                json_fixer = half_json.JSONFixer()
                fix_resp = json_fixer.fix(response_text)
                if fix_resp.success:
                    print("fix success, use fixed response_text: ", fix_resp.line)
                    response_json = json.loads(fix_resp.line)
                else:
                    raise e

            mz_prompt_utils.Utils.print_log(
                f"response_json: {json.dumps(response_json, indent=2)}")

            responses = []
            for key, value in response_json.items():
                if type(value) == list:
                    # 去除开头.和空格
                    value = [v.strip().lstrip(".") for v in value]
                    # 去除空字符串
                    value = [v for v in value if v != ""]
                    if len(value) > 0:
                        responses.append(f"({', '.join(value)})")

                else:
                    if value != "":
                        responses.append(f"({value})")

            response = ", ".join(responses)
        else:
            response = mz_llama_cpp.llama_cpp_simple_interrogator(
                model_file=model_file,
                system=system_prompt,
                question=question,
                options=options,
            )

            start_str = customize_instruct.get("start_str", "")
            if start_str != "" and response.find(start_str) != -1:
                full_response_list = response.split(start_str)
                # 删除第一个元素
                full_response_list.pop(0)
                response = start_str.join(full_response_list)

            end_str = customize_instruct.get("end_str", "")
            if end_str != "" and response.find(end_str) != -1:
                full_response_list = response.split(end_str)
                # 删除最后一个元素
                full_response_list.pop()
                response = end_str.join(full_response_list)

        if keep_device is False:
            mz_llama_cpp.freed_gpu_memory(model_file=model_file)

        # 去除换行
        while response.find("\n") != -1:
            response = response.replace("\n", " ")

        # 句号换成逗号
        while response.find(".") != -1:
            response = response.replace(".", ",")

        # 去除多余逗号
        while response.find(",,") != -1:
            response = response.replace(",,", ",")
        while response.find(", ,") != -1:
            response = response.replace(", ,", ",")

        if translate_option is True:
            response = mz_prompt_utils.Utils.prompt_zh_to_en(response)

        style_presets_prompt_text = style_presets_prompt.get(style_presets, "")

        if style_presets_prompt_text != "":
            response = f"{style_presets_prompt_text}, {response}"

    except Exception as e:
        mz_llama_cpp.freed_gpu_memory(model_file=model_file)
        raise e

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = mz_prompt_utils.Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [mz_prompt_utils.Utils.to_debug_prompt(response, translate_option),]}, "result": (response, conditionings)}


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
        onec_args_dict["image"] = mz_prompt_utils.Utils.pil2tensor(pil_image)

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
                prompt_fixed_beginning = captioner_config.get(
                    "prompt_fixed_beginning", "")
                f.write(prompt_fixed_beginning + response)

        result.append(response)

        # mz_prompt_webserver.show_toast_success(
        #     f"提示词保存成功(prompt saved successfully): {caption_file}",
        #     1000,
        # )

    return result


def image_interrogator_node_encode(args_dict):
    importlib.reload(mz_prompts)

    captioner_config = args_dict.get("captioner_config", None)
    if captioner_config is not None:
        image_interrogator_captioner(args_dict)
        # raise Exception(
        #     "图片批量反推任务已完成 ; Image batch reverse push task completed")
        return {"ui": {"string": ["图片批量反推任务已完成 ; Image batch reverse push task completed",]}, "result": ("", None)}

    model_config = args_dict.get("image_interrogator_model", {})

    chat_format = model_config.get("chat_format", None)
    llama_cpp_model = model_config.get("llama_cpp_model", "auto")
    mmproj_model = model_config.get("mmproj_model", "auto")

    select_model_type = model_config.get("type", "ManualSelect")
    if select_model_type == "ManualSelect":
        llama_cpp_model = model_config.get("model_path", "auto")
        if llama_cpp_model == "auto":
            llama_cpp_model = mz_prompt_utils.Utils.get_auto_model_fullpath(
                "ggml_llava1_5-7b-q4_k_m")
        else:
            llama_cpp_model = os.path.join(
                mz_prompt_utils.Utils.get_gguf_models_path(), llama_cpp_model)

        if mmproj_model.endswith("auto"):
            llama_cpp_model_sha256 = mz_prompt_utils.Utils.file_sha256(
                llama_cpp_model)

            mmproj_model_name = mz_prompt_utils.Utils.get_model_zoo(
                tags_filter=llama_cpp_model_sha256)
            if len(mmproj_model_name) == 0:
                mmproj_model_name = None
            else:
                mmproj_model_name = mmproj_model_name[0].get("model", None)

            if mmproj_model_name is None:
                mz_prompt_utils.Utils.print_log(
                    "llama_cpp_model_sha256: ", llama_cpp_model_sha256)
                raise Exception(
                    "未能自动找到对应的mmproj文件 ; Failed to automatically find the corresponding mmproj file.")
            else:
                pass

            mmproj_model = mz_prompt_utils.Utils.get_auto_model_fullpath(
                mmproj_model_name)
        else:
            # mmproj_model = os.path.join(
            #     mz_prompt_utils.Utils.get_gguf_models_path(), mmproj_model)
            pass

    elif select_model_type == "DownloaderSelect":
        model_name = model_config.get("model_name")
        llama_cpp_model = mz_prompt_utils.Utils.get_auto_model_fullpath(
            model_name)

        mmproj_model = model_config.get("mmproj_model_name", "auto")

        mmproj_model_name = mmproj_model
        if mmproj_model == "auto":
            llama_cpp_model_sha256 = mz_prompt_utils.Utils.file_sha256(
                llama_cpp_model)

            mz_prompt_utils.Utils.print_log(
                "llama_cpp_model_sha256: ", llama_cpp_model_sha256)

            mmproj_model_name = mz_prompt_utils.Utils.get_model_zoo(
                tags_filter=llama_cpp_model_sha256)
            if len(mmproj_model_name) == 0:
                mmproj_model_name = None
            else:
                mmproj_model_name = mmproj_model_name[0].get("model", None)

            if mmproj_model_name is None:
                raise Exception(
                    "未能自动找到对应的mmproj文件 ; Failed to automatically find the corresponding mmproj file")

        mmproj_model = mz_prompt_utils.Utils.get_auto_model_fullpath(
            mmproj_model_name)

    else:
        raise Exception("Unknown select_model_type")

    image = args_dict.get("image", None)
    image = mz_prompt_utils.Utils.tensor2pil(image)

    resolution = args_dict.get("resolution", 512)
    keep_device = args_dict.get("keep_device", False)
    seed = args_dict.get("seed", -1)
    options = args_dict.get("llama_cpp_options", {})
    options["seed"] = seed
    options["chat_format"] = chat_format

    image = mz_prompt_utils.Utils.resize_max(image, resolution, resolution)

    customize_instruct = args_dict.get("customize_instruct", None)
    if customize_instruct is None:
        # system_prompt = mz_prompts.GPT4VImageCaptioner_System
        # question = mz_prompts.GPT4VImageCaptioner_Prompt

        # system_prompt = mz_prompts.M_ImageCaptioner2_System
        # question = mz_prompts.M_ImageCaptioner2_Prompt

        system_prompt = "You are an assistant who perfectly describes images."
        question = "Describe this image in detail please."
    else:
        system_prompt = customize_instruct.get("system", "")
        question = customize_instruct.get("instruct", "")

    mz_prompt_utils.Utils.print_log(f"mmproj_model: {mmproj_model}")
    response = mz_llama_cpp.llava_cpp_simple_interrogator(
        model_file=llama_cpp_model,
        mmproj_file=mmproj_model,
        image=image,
        options=options,
        system=system_prompt,
        question=question,
    )
    response = response.strip()
    if response is not None and response != "":

        if args_dict.get("post_processing", False):

            # 双引号换成空格
            response = response.replace("\"", " ")
            # 中括号换成空格
            response = response.replace("[", " ")
            response = response.replace("]", " ")

            # 括号换成空格
            response = response.replace("(", " ")
            response = response.replace(")", " ")

            # 去除多余空格
            while response.find("  ") != -1:
                response = response.replace("  ", " ")

            # 从第一个为英文字母的地方开始截取
            for i in range(len(response)):
                if response[i].isalpha():
                    response = response[i:]
                    break

            response = response.strip()
            schema = get_schema_obj(
                keys_type={
                    "short_describes": get_schema_base_type("string"),
                    "subject_tags": get_schema_array("string"),
                    "action_tags": get_schema_array("string"),
                    "light_tags": get_schema_array("string"),
                    "scene_tags": get_schema_array("string"),
                    "mood_tags": get_schema_array("string"),
                    "style_tags": get_schema_array("string"),
                    "object_tags": get_schema_array("string"),
                },
                required=[
                    "short_describes",
                    "subject_tags",
                    "action_tags",
                    "lights_tags",
                    "scenes_tags",
                    "moods_tags",
                    "styles_tags",
                    "objects_tags",
                ]
            )
            response_json_str = mz_llama_cpp.llama_cpp_simple_interrogator_to_json(
                model_file=llama_cpp_model,
                system=mz_prompts.ImageCaptionerPostProcessing_System,
                question=f"Content: {response}",
                schema=schema,
                options=options,
            )

            try:
                response_json = json.loads(response_json_str)
            except Exception as e:
                from . import half_json
                print("json.loads failed, try fix response_json_str: ",
                      response_json_str)
                json_fixer = half_json.JSONFixer()
                fix_resp = json_fixer.fix(response_json_str)
                if fix_resp.success:
                    print("fix success, use fixed response_json_str: ",
                          fix_resp.line)
                    response_json = json.loads(fix_resp.line)
                else:
                    raise e

            responses = []

            def pure_words(text: str) -> bool:
                number_of_spaces = text.count(" ")
                if number_of_spaces > 2:
                    return False
                for c in text:
                    if not c.isalpha() and c != "-" and c != "_" and c != " ":
                        return False

                return True

            for key, value in response_json.items():
                if type(value) == list:

                    # 去除开头.和空格
                    value = [v.strip().lstrip(".") for v in value]
                    # 去除空字符串
                    value = [v for v in value if v != ""]

                    # 去除带有空格和标点符号的字符串
                    value = [
                        v for v in value if pure_words(v)]

                    # 空格换成下划线
                    value = [v.replace(" ", "_") for v in value]

                    # 首字母小写
                    value = [v.lower() for v in value]

                    if len(value) > 0:
                        responses.append(f"{', '.join(value)}")

            description = response_json.get("short_describes", "")
            if description != "":
                responses.append(f"{description}")

            # 对response进行去重
            response = ", ".join(responses)

    if keep_device is False:
        mz_llama_cpp.freed_gpu_memory(model_file=llama_cpp_model)

    # return response

    conditionings = None
    clip = args_dict.get("clip", None)
    if clip is not None:
        conditionings = mz_prompt_utils.Utils.a1111_clip_text_encode(
            clip, response, )

    return {"ui": {"string": [mz_prompt_utils.Utils.to_debug_prompt(response),]}, "result": (response, conditionings)}
