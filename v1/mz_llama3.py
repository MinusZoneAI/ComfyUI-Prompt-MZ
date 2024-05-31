import json
from .. import mz_prompt_utils
from .. import mz_llama_cpp
from . import mz_deprecated
import importlib


llama3_models = [
    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
    "Meta-Llama-3-8B-Instruct.Q3_K_L.gguf",
    "Meta-Llama-3-8B-Instruct.Q3_K_M.gguf",
    "Meta-Llama-3-8B-Instruct.Q3_K_S.gguf",
    "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "Meta-Llama-3-8B-Instruct.Q4_1.gguf",
    "Meta-Llama-3-8B-Instruct.Q4_K_S.gguf",
    "Meta-Llama-3-8B-Instruct.Q5_0.gguf",
    "Meta-Llama-3-8B-Instruct.Q5_1.gguf",
    "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
    "Meta-Llama-3-8B-Instruct.Q5_K_S.gguf",
    "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
    "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
]


def get_exist_model(model_name):
    modelscope_model_path = mz_prompt_utils.Utils.modelscope_download_model(
        model_type="llama3",
        model_name=model_name,
        only_get_path=True,
    )

    if modelscope_model_path is not None:
        return modelscope_model_path

    model_url = f"https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
    hf_model_path = mz_prompt_utils.Utils.hf_download_model(
        model_url, only_get_path=True)
    if hf_model_path is not None:
        return hf_model_path

    return None


def query_beautify_prompt_text(args_dict):
    model_name = args_dict.get("llama_cpp_model", "")
    download_source = args_dict.get("download_source", None)

    try:
        model_file = get_exist_model(model_name)

        if model_file is None:
            if download_source == "modelscope":
                model_file = mz_prompt_utils.Utils.modelscope_download_model(
                    model_type="llama3",
                    model_name=model_name,
                )
            else:
                model_url = f"https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
                if download_source == "hf-mirror.com":
                    model_url = f"https://hf-mirror.com/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model_name}"
                model_file = mz_prompt_utils.Utils.hf_download_model(model_url)

        args_dict["llama_cpp_model"] = model_file
        full_response = mz_deprecated.base_query_beautify_prompt_text(
            args_dict=args_dict)
        return full_response

    except Exception as e:
        mz_llama_cpp.freed_gpu_memory(model_file=model_file)
        # mz_utils.Utils.print_log(f"Error in auto_prompt_text: {e}")
        raise e
