

import os
import sys
from .mz_prompt_utils import Utils  
from nodes import MAX_RESOLUTION
import comfy.utils 
import shutil
import comfy.samplers 

AUTHOR_NAME = u"[MinusZone]" 
CATEGORY_NAME = f"{AUTHOR_NAME} Utils"

sys.path.append(os.path.join(os.path.dirname(__file__))) 

import importlib



NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = { 
}

  
import mz_llama3
import mz_llama_cpp
import mz_llava





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

class MZ_LLamaCPPOptions:
    @classmethod
    def INPUT_TYPES(s):
        opt = mz_llama_cpp.LlamaCppOptions()
        value = opt.value
        result = {}

        for key in value:
            if type(value[key]) == bool:
                result[key] = ([True, False], {"default": value[key]})
            elif type(value[key]) == int:
                result[key] = ("INT", {"default": value[key], "min": 0, "max": 0xffffffffffffffff})
            elif type(value[key]) == float:
                result[key] = ("FLOAT", {"default": value[key], "min": 0, "max": 0xffffffffffffffff})
            elif type(value[key]) == str:
                result[key] = ("STRING", {"default": value[key]})
            else:
                raise Exception(f"Unknown type: {type(value[key])}")
        
        return {
            "required": result,
        }
    

    RETURN_TYPES = ("LLamaCPPOptions",)
    FUNCTION = "create"
    CATEGORY = CATEGORY_NAME
    def create(self, **kwargs):
        importlib.reload(mz_llama_cpp)
        opt = mz_llama_cpp.LlamaCppOptions()
        for key in kwargs:
            opt.value[key] = kwargs[key]
        return (opt,)

NODE_CLASS_MAPPINGS["MZ_LLamaCPPOptions"] = MZ_LLamaCPPOptions
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLamaCPPOptions"] = f"{AUTHOR_NAME} - LLamaCPPOptions"

class MZ_LLama3CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        m_llama3_models = llama3_models.copy()
        for i in range(len(m_llama3_models)):
            if mz_llama3.get_exist_model(m_llama3_models[i]) is not None:
                m_llama3_models[i] += "[downloaded]"

        
        importlib.reload(mz_llama3)
        style_presets = mz_llama3.get_style_presets()

        return {
            "required": {
                "llama_cpp_model": (m_llama3_models, {"default": m_llama3_models[0]}),
                "download_source": (
                    ["none", "modelscope", "hf-mirror.com",],
                    {"default": "none"}
                ),
                "style_presets": (
                    style_presets, {"default": style_presets[0]}
                ),
                "text": ("STRING", {"multiline": True,}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "n_gpu_layers": ("INT", {"default": 40, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "clip": ("CLIP", ),
                "llama_cpp_options": ("LLamaCPPOptions", ),
            },
        }
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME
    def encode(self, text, llama_cpp_model, style_presets, clip=None, download_source=None, seed=0, n_gpu_layers=40, llama_cpp_options=None):
        importlib.reload(mz_llama3)

        llama_cpp_model = llama_cpp_model.replace("[downloaded]", "")

        options = {}
        if llama_cpp_options is not None:
            options = llama_cpp_options.value
        text = mz_llama3.query_beautify_prompt_text(llama_cpp_model, n_gpu_layers, text, style_presets, download_source, options) 
        conditionings = None
        if clip is not None:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]
 
        return (text, conditionings)
    
NODE_CLASS_MAPPINGS["MZ_LLama3CLIPTextEncode"] = MZ_LLama3CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLama3CLIPTextEncode"] = f"{AUTHOR_NAME} - LLama3CLIPTextEncode"


LLava_models = [ 
    "ggml_bakllava-1/ggml-model-q4_k.gguf",
    "ggml_bakllava-1/ggml-model-q5_k.gguf",
    "ggml_bakllava-1/ggml-model-f16.gguf",
    "ggml_llava-v1.5-7b/ggml-model-q4_k.gguf",
    "ggml_llava-v1.5-7b/ggml-model-q5_k.gguf",
    "ggml_llava-v1.5-7b/ggml-model-f16.gguf",
]

LLava_mmproj_models = [
    # "llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf",
    "ggml_bakllava-1/mmproj-model-f16.gguf",
    "ggml_llava-v1.5-7b/mmproj-model-f16.gguf",
]

class MZ_LLavaImageInterrogator:
    @classmethod
    def INPUT_TYPES(s): 

        m_llava_models = LLava_models.copy()
        for i in range(len(m_llava_models)):
            if mz_llava.get_exist_model(m_llava_models[i]) is not None:
                m_llava_models[i] += "[downloaded]"

        m_llava_mmproj_models = LLava_mmproj_models.copy()
        for i in range(len(m_llava_mmproj_models)):
            if mz_llava.get_exist_model(m_llava_mmproj_models[i]) is not None:
                m_llava_mmproj_models[i] += "[downloaded]"


        return {"required": {
            "llama_cpp_model": (m_llava_models, {"default": m_llava_models[0]}),
            "mmproj_model": (m_llava_mmproj_models, {"default": m_llava_mmproj_models[0]}),
            "download_source": (
                [
                    "none",  
                    "modelscope", 
                    "hf-mirror.com",
                ],
                {"default": "none"}
            ),
            "image":  ("IMAGE",),
            "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
            # "prefix" : ("STRING", {"default": "(masterpiece)", "multiline": True, }),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "n_gpu_layers": ("INT", {"default": 40, "min": -1, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "clip": ("CLIP", ),
            "llama_cpp_options": ("LLamaCPPOptions", ),
        }}
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    FUNCTION = "interrogate"
    CATEGORY = CATEGORY_NAME
    def interrogate(self, llama_cpp_model, mmproj_model, image, resolution, download_source=None, seed=0, clip=None, n_gpu_layers=40, llama_cpp_options=None):
        importlib.reload(mz_llava)

        llama_cpp_model = llama_cpp_model.replace("[downloaded]", "")
        mmproj_model = mmproj_model.replace("[downloaded]", "")

        # prefix = Utils.prompt_zh_to_en(prefix)
        # if not prefix.endswith(","):
        #     prefix += ","

        image_pil = Utils.tensor2pil(image)


        options = {}
        if llama_cpp_options is not None:
            options = llama_cpp_options.value

        response = mz_llava.image_interrogator(
            llama_cpp_model,
            mmproj_model,
            n_gpu_layers,
            image_pil, 
            resolution, 
            download_source,
            options,
        ) 
        conditionings = None
        if clip is not None:
            tokens = clip.tokenize(response)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]
        return (response, conditionings)

NODE_CLASS_MAPPINGS["MZ_LLavaImageInterrogator"] = MZ_LLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLavaImageInterrogator"] = f"{AUTHOR_NAME} - LLavaImageInterrogator"

 
class MZ_LLamaCPPInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_file": ("STRING", {"default": ""}),
                "use_system": ([True, False], {"default": True}),
                "system": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True,}),
                "n_gpu_layers": ("INT", {"default": 40, "min": -1, "max": 0xffffffffffffffff}),
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "simple_interrogator"
    CATEGORY = CATEGORY_NAME
    def simple_interrogator(self, model_file, prompt, use_system=True, system="You are a helpful assistant.", n_gpu_layers=40):
        importlib.reload(mz_llama_cpp)
        result = mz_llama_cpp.llama_cpp_simple_interrogator(
            model_file, 
            n_gpu_layers=n_gpu_layers,
            use_system=use_system,
            system=system,
            question=prompt,
        )
        return (result,)
    
NODE_CLASS_MAPPINGS["MZ_LLamaCPPInterrogator"] = MZ_LLamaCPPInterrogator 
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLamaCPPInterrogator"] = f"{AUTHOR_NAME} - LLamaCPP simple interrogator"