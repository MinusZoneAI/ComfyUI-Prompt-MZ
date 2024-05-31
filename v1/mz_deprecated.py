from ..mz_prompt_utils import Utils  
from ..mz_llama_cpp import *
from ..mz_llama_core_nodes import *
from ..mz_prompts import *

def base_query_beautify_prompt_text(args_dict):
    model_file = args_dict.get("llama_cpp_model", "")
    text = args_dict.get("text", "")
    style_presets = args_dict.get("style_presets", "")
    options = args_dict.get("llama_cpp_options", {})
    keep_device = args_dict.get("keep_device", False)
    seed = args_dict.get("seed", -1)
    options["seed"] = seed
  

    customize_instruct = args_dict.get("customize_instruct", None)
    Utils.print_log(
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

            Utils.print_log(f"system_prompt: {system_prompt}")
            Utils.print_log(f"question: {question}")

        if schema is not None:
            response_json = llama_cpp_simple_interrogator_to_json(
                model_file=model_file,
                system=system_prompt,
                question=question,
                schema=schema,
                options=options,
            )
            Utils.print_log(f"response_json: {response_json}")

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
            full_response = llama_cpp_simple_interrogator(
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
            freed_gpu_memory(model_file=model_file)

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

        full_response = Utils.prompt_zh_to_en(full_response)

        style_presets_prompt_text = style_presets_prompt.get(style_presets, "")

        if style_presets_prompt_text != "":
            full_response = f"{style_presets_prompt_text}, {full_response}"

        return full_response

    except Exception as e:
        freed_gpu_memory(model_file=model_file)
        # mz_utils.Utils.print_log(f"Error in auto_prompt_text: {e}")
        raise e
