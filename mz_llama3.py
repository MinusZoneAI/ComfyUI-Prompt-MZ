import json
import mz_utils  
import mz_llama_cpp

import importlib



def get_exist_model(model_name):
    modelscope_model_path = mz_utils.Utils.modelscope_download_model(
        model_type="llama3",
        model_name=model_name,
        only_get_path=True,
    )

    if modelscope_model_path is not None:
        return modelscope_model_path

    model_url = f"https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
    hf_model_path = mz_utils.Utils.hf_download_model(model_url, only_get_path=True)
    if hf_model_path is not None:
        return hf_model_path

    return None

high_quality_prompt = "((high quality:1.4), (best quality:1.4), (masterpiece:1.4), (8K resolution), (2k wallpaper))"
style_presets_prompt = {
    "high_quality": high_quality_prompt,
    "photography": f"{high_quality_prompt}, (RAW photo, best quality), (realistic, photo-realistic:1.2), (bokeh, cinematic shot, dynamic composition, incredibly detailed, sharpen, details, intricate detail, professional lighting, film lighting, 35mm, anamorphic, lightroom, cinematography, bokeh, lens flare, film grain, HDR10, 8K)",
    "illustration": f"{high_quality_prompt}, ((detailed matte painting, intricate detail, splash screen, complementary colors), (detailed),(intricate details),illustration,an extremely delicate and beautiful,ultra-detailed,highres,extremely detailed)",
}
def get_style_presets():
    return [
        "high_quality",
        "photography",
        "illustration",  
    ]

def query_beautify_prompt_text(model_name, n_gpu_layers, text, style_presets, download_source=None):     
    import mz_prompts
    importlib.reload(mz_prompts)
    importlib.reload(mz_llama_cpp)



    try: 
        model_file = get_exist_model(model_name)
        
        if model_file is None:
            if download_source == "modelscope":
                model_file = mz_utils.Utils.modelscope_download_model(
                    model_type="llama3",
                    model_name=model_name,
                )
            else:
                model_url = f"https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
                if download_source == "hf-mirror.com":
                    model_url = f"https://hf-mirror.com/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
                model_file = mz_utils.Utils.hf_download_model(model_url)
                
        schema = mz_llama_cpp.get_schema_obj(
            keys_type={
                "description": mz_llama_cpp.get_schema_base_type("string"),
                "long_prompt": mz_llama_cpp.get_schema_base_type("string"),
                "main_color_word": mz_llama_cpp.get_schema_base_type("string"),
                "camera_angle_word": mz_llama_cpp.get_schema_base_type("string"),
                "style_words": mz_llama_cpp.get_schema_array("string"),
                "subject_words": mz_llama_cpp.get_schema_array("string"),
                "light_words": mz_llama_cpp.get_schema_array("string"),
                "environment_words": mz_llama_cpp.get_schema_array("string"),
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

        response_json = mz_llama_cpp.llama_cpp_simple_interrogator_to_json(
            model_file=model_file,
            n_gpu_layers=n_gpu_layers,
            system=mz_prompts.Beautify_Prompt,
            question=f"IDEA: {style_presets},{text}",
            schema=schema,
            temperature=1.6,
            max_tokens=2048,
        ) 
        
        mz_llama_cpp.freed_gpu_memory(model_file=model_file)
        
        
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
        if len(response["style_words"]) > 0:
            response["style_words"] = [x for x in response["style_words"] if x != ""]
            full_responses.append(f"({', '.join(response['style_words'])})")
        if len(response["subject_words"]) > 0:
            response["subject_words"] = [x for x in response["subject_words"] if x != ""]
            full_responses.append(f"({', '.join(response['subject_words'])})")
        if len(response["light_words"]) > 0:
            response["light_words"] = [x for x in response["light_words"] if x != ""]
            full_responses.append(f"({', '.join(response['light_words'])})")
        if len(response["environment_words"]) > 0:
            response["environment_words"] = [x for x in response["environment_words"] if x != ""]
            full_responses.append(f"({', '.join(response['environment_words'])})")

        full_response = ", ".join(full_responses)


        
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

        full_response = mz_utils.Utils.prompt_zh_to_en(full_response) 


        style_presets_prompt_text = style_presets_prompt.get(style_presets, "")
        full_response = f"{style_presets_prompt_text}, {full_response}"
        return full_response

    except Exception as e:
        # mz_utils.Utils.print_log(f"Error in auto_prompt_text: {e}")
        raise e