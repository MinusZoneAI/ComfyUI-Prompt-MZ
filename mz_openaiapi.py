import os
import sys
import json
import subprocess

from . import mz_prompt_utils
from . import mz_llama_cpp
from . import mz_llama_core_nodes
from . import mz_prompts


def zhipu_json_fix(input_data):
    if type(input_data) == dict:
        if "Items" in input_data:
            return input_data["Items"]
        else:
            for key, value in input_data.items():
                input_data[key] = zhipu_json_fix(value)
            return input_data

    elif type(input_data) == list:
        for i in range(len(input_data)):
            input_data[i] = zhipu_json_fix(input_data[i])
        return input_data

    else:
        return input_data


def query_beautify_prompt_text(args_dict):
    try:
        from openai import OpenAI
        import openai
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai"])
        from openai import OpenAI
        import openai

    api_key = args_dict.get("api_key", None)
    base_url = args_dict.get("base_url", None)

    text = args_dict.get("text", "")
    style_presets = args_dict.get("style_presets", "")

    if api_key is None:
        raise ValueError("api_key is required")

    client = OpenAI(
        api_key=api_key,
        default_headers={"x-foo": "true"}
    )

    if base_url is not None:
        client.base_url = base_url

    model_name = args_dict.get("model_name", "gpt-3.5-turbo")

    options = args_dict.get("options", {})

    customize_instruct = args_dict.get("customize_instruct", None)
    mz_prompt_utils.Utils.print_log(
        f"customize_instruct: {customize_instruct}")

    schema = None
    if customize_instruct is None:
        schema = mz_llama_core_nodes.get_schema_obj(
            keys_type={
                "description": mz_llama_core_nodes.get_schema_base_type("string"),
                "long_prompt": mz_llama_core_nodes.get_schema_base_type("string"),
                "main_color_word": mz_llama_core_nodes.get_schema_base_type("string"),
                "camera_angle_word": mz_llama_core_nodes.get_schema_base_type("string"),
                "style_words": mz_llama_core_nodes.get_schema_array("string"),
                "subject_words": mz_llama_core_nodes.get_schema_array("string"),
                "light_words": mz_llama_core_nodes.get_schema_array("string"),
                "environment_words": mz_llama_core_nodes.get_schema_array("string"),
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
        # print(f"system_prompt: {system_prompt}")
        # print(f"question: {question}")

    output = None
    if schema is not None:

        output = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{question}\ncall beautify_prompt_text function to get the result."},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "beautify_prompt_text",
                    "description": "required Beautify Prompt Text",
                    "parameters": schema,
                }
            }],
            tool_choice={"type": "function",
                         "function": {"name": "beautify_prompt_text"}},
        )
        tool_calls = output.choices[0].message.tool_calls

        functions_args = {}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            functions_args[function_name] = function_args
        beautify_prompt_text_result = functions_args.get(
            "beautify_prompt_text", {})

        mz_prompt_utils.Utils.print_log(
            f"beautify_prompt_text_result: {beautify_prompt_text_result}")

        beautify_prompt_text_result = zhipu_json_fix(
            beautify_prompt_text_result)
        results = []
        for key, value in beautify_prompt_text_result.items():
            if type(value) == list:
                value = [item for item in value if item != ""]
                value = [mz_prompt_utils.Utils.prompt_zh_to_en(item)
                         for item in value]
                if len(value) == 0:
                    continue
                item_str = ", ".join(value)
                results.append(f"({item_str})")
            else:
                if value == "":
                    continue
                value = mz_prompt_utils.Utils.prompt_zh_to_en(value)
                results.append(f"({value})")

        full_response = ", ".join(results)

    else:
        output = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        full_response = output.choices[0].message.content

    mz_prompt_utils.Utils.print_log(
        f"OPENAI_OUTPUT: \n{output.model_dump_json()}")
    # print(output.model_dump_json())

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
    style_presets_prompt_text = mz_llama_core_nodes.style_presets_prompt.get(
        style_presets, "")
    if style_presets_prompt_text != "":
        full_response = f"{style_presets_prompt_text}, {full_response}"
    return full_response
