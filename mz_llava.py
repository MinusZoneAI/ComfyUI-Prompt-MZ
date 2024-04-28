import json
import os
import mz_prompt_utils  
import mz_llama_cpp

import importlib

import mz_prompts


LLava_models = [ 
    "llava-1.6-mistral-7b-gguf/llava-v1.6-mistral-7b.Q5_K_M.gguf",
    "llava-v1.6-vicuna-13b-gguf/llava-v1.6-vicuna-13b.Q5_K_M.gguf",
    "ggml_llava-v1.5-7b/ggml-model-q4_k.gguf", 
    "ggml_llava-v1.5-7b/ggml-model-q5_k.gguf",
    "ggml_llava-v1.5-7b/ggml-model-f16.gguf",
    "ggml_bakllava-1/ggml-model-q4_k.gguf",
    "ggml_bakllava-1/ggml-model-q5_k.gguf",
    "ggml_bakllava-1/ggml-model-f16.gguf",
]

LLava_mmproj_models = [
    "llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf",
    "llava-v1.6-vicuna-13b-gguf/mmproj-model-f16.gguf",
    "ggml_llava-v1.5-7b/mmproj-model-f16.gguf",
    "ggml_bakllava-1/mmproj-model-f16.gguf",
]


huggingface_models_map = {
    "llava-v1.6-vicuna-13b-gguf": "cjpais",
    "llava-1.6-mistral-7b-gguf": "cjpais",
    "ggml_llava-v1.5-7b": "mys",
    "ggml_llava-v1.5-13b": "mys",
    "ggml_bakllava-1": "mys",
}


def get_exist_model(model_name):
    modelscope_model_path = mz_prompt_utils.Utils.modelscope_download_model(
        model_type="llava",
        model_name=model_name,
        only_get_path=True,
    )

    if modelscope_model_path is not None:
        return modelscope_model_path
    

    model_name = model_name.split("?")[0]
    model_names = model_name.split("/")
    
    author = huggingface_models_map.get(model_names[0], None)
    if author is None:
        return False

    model_url = f"https://hf-mirror.com/{author}/{model_names[0]}/resolve/main/{model_names[1]}"

    
    hf_model_path = mz_prompt_utils.Utils.hf_download_model(model_url, only_get_path=True)

    if hf_model_path is not None:
        return hf_model_path
    
    
    return None


def image_interrogator(args_dict):  
    model_name = args_dict.get("llama_cpp_model", "")
    mmproj_name = args_dict.get("mmproj_model", "")
    download_source = args_dict.get("download_source", None)  
    model_file = get_exist_model(model_name) 
    mmproj_file = get_exist_model(mmproj_name) 
    
    if model_file is None or mmproj_file is None:
        if download_source == "modelscope":
            if model_file is None:
                model_file = mz_prompt_utils.Utils.modelscope_download_model(
                    model_type="llava",
                    model_name=model_name,
                )
            if mmproj_file is None:
                mmproj_file = mz_prompt_utils.Utils.modelscope_download_model(
                    model_type="llava",
                    model_name=mmproj_name,
                )
        else:
            model_name = model_name.split("?")[0]
            model_names = model_name.split("/")

            author = huggingface_models_map.get(model_names[0], None)
            if author is None:
                raise Exception(f"Model {model_names[0]} is not supported for image_to_text.")
            
            if download_source == "hf-mirror.com":
                model_url = f"https://hf-mirror.com/{author}/{model_names[0]}/resolve/main/{model_names[1]}" 
            else:
                model_url = f"https://huggingface.co/{author}/{model_names[0]}/resolve/main/{model_names[1]}"


            if model_file is None:
                model_file = mz_prompt_utils.Utils.hf_download_model(model_url)

            mmproj_name = mmproj_name.split("?")[0]
            mmproj_names = mmproj_name.split("/")
            if download_source == "hf-mirror.com":
                mmproj_url = f"https://hf-mirror.com/{author}/{mmproj_names[0]}/resolve/main/{mmproj_names[1]}" 
            else:   
                mmproj_url = f"https://huggingface.co/{author}/{mmproj_names[0]}/resolve/main/{mmproj_names[1]}"

            if mmproj_file is None:
                mmproj_file = mz_prompt_utils.Utils.hf_download_model(mmproj_url)



    args_dict["llama_cpp_model"] = model_file
    args_dict["mmproj_model"] = mmproj_file
    response = base_image_interrogator(args_dict=args_dict)
    return response
        

def base_image_interrogator(args_dict):
    model_file = args_dict.get("llama_cpp_model", "")
    mmproj_file = args_dict.get("mmproj_model", "")
    image = args_dict.get("image", None)
    resolution = args_dict.get("resolution", 512)
    keep_device = args_dict.get("keep_device", False)
    seed = args_dict.get("seed", -1) 
    options = args_dict.get("llama_cpp_options", {})
    options["seed"] = seed
    
    
    image = mz_prompt_utils.Utils.resize_max(image, resolution, resolution)
 
    
 


    response = mz_llama_cpp.llava_cpp_simple_interrogator(
        model_file=model_file,
        mmproj_file=mmproj_file, 
        image=image,
        options=options,
    )

 

    sd_format = args_dict.get("sd_format", "v1")
    if sd_format == "v1":
        
        mz_prompt_utils.Utils.print_log(f"response: {response}")
    
        
        schema = mz_llama_cpp.get_schema_obj( 
            keys_type={
                "short_describes":  mz_llama_cpp.get_schema_base_type("string"),   
                "subject_tags":  mz_llama_cpp.get_schema_array("string"),
                "action_tags":  mz_llama_cpp.get_schema_array("string"),
                "light_tags":  mz_llama_cpp.get_schema_array("string"), 
                "scenes_tags":  mz_llama_cpp.get_schema_array("string"),
                "other_tags":  mz_llama_cpp.get_schema_array("string"),
            },
            required=[
                "short_describes", 
                "subject_tags",
                "action_tags",
                "light_tags",
                "scenes_tags",
                "other_tags",
            ]
        ) 
        
        response = mz_llama_cpp.llama_cpp_simple_interrogator_to_json(
            model_file=model_file, 
            system=mz_prompts.Beautify_Prompt,
            question=f"IDEA: {response}",
            schema=schema,
            options=options, 
        )


        response_json = json.loads(response)
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

 
    
    if keep_device is False:
        mz_llama_cpp.freed_gpu_memory(model_file=model_file)
    return response
        
