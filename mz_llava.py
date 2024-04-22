import json
import os
import mz_utils  
import mz_llama_cpp

import importlib



huggingface_models_map = {
    "llava-1.6-mistral-7b-gguf": "cjpais",
    "ggml_llava-v1.5-7b": "mys",
    "ggml_llava-v1.5-13b": "mys",
    "ggml_bakllava-1": "mys",
}


def get_exist_model(model_name):
    modelscope_model_path = mz_utils.Utils.modelscope_download_model(
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

    
    hf_model_path = mz_utils.Utils.hf_download_model(model_url, only_get_path=True)

    if hf_model_path is not None:
        return hf_model_path
    
    
    return None


def image_interrogator(model_name, mmproj_name, image, resolution, download_source=None):  
    image = mz_utils.Utils.resize_max(image, resolution, resolution)
    model_file = get_exist_model(model_name)
    mmproj_file = get_exist_model(mmproj_name)
    
    if model_file is None or mmproj_file is None:
        if download_source == "modelscope":
            if model_file is None:
                model_file = mz_utils.Utils.modelscope_download_model(
                    model_type="llava",
                    model_name=model_name,
                )
            if mmproj_file is None:
                mmproj_file = mz_utils.Utils.modelscope_download_model(
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
                model_file = mz_utils.Utils.hf_download_model(model_url)

            mmproj_name = mmproj_name.split("?")[0]
            mmproj_names = mmproj_name.split("/")
            if download_source == "hf-mirror.com":
                mmproj_url = f"https://hf-mirror.com/{author}/{mmproj_names[0]}/resolve/main/{mmproj_names[1]}" 
            else:   
                mmproj_url = f"https://huggingface.co/{author}/{mmproj_names[0]}/resolve/main/{mmproj_names[1]}"

            if mmproj_file is None:
                mmproj_file = mz_utils.Utils.hf_download_model(mmproj_url)


    response = mz_llama_cpp.llava_cpp_simple_interrogator(
        model_file=model_file,
        mmproj_file=mmproj_file,
        image=image,
    )
 
    return response
        
