

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
import mz_phi3
import mz_llama_cpp
import mz_llava


class MZ_LLamaCPPOptions:
    @classmethod
    def INPUT_TYPES(s):
        value = mz_llama_cpp.LlamaCppOptions()
        result = {}

        for key in value:
            if type(value[key]) == bool:
                result[key] = ([True, False], {"default": value[key]})
            elif type(value[key]) == int:
                result[key] = ("INT", {
                               "default": value[key], "min": -0xffffffffffffffff, "max": 0xffffffffffffffff})
            elif type(value[key]) == float:
                result[key] = ("FLOAT", {
                               "default": value[key], "min": -0xffffffffffffffff, "max": 0xffffffffffffffff})
            elif type(value[key]) == str:
                result[key] = ("STRING", {"default": value[key]})
            elif type(value[key]) == list:
                result[key] = (value[key], {"default": value[key][0]})
            else:
                raise Exception(f"Unknown type: {type(value[key])}")

        return {
            "required": result,
        }

    RETURN_TYPES = ("LLamaCPPOptions",)
    RETURN_NAMES = ("llama_cpp_options",)

    FUNCTION = "create"
    CATEGORY = CATEGORY_NAME

    def create(self, **kwargs):
        importlib.reload(mz_llama_cpp)
        opt = {}
        for key in kwargs:
            opt[key] = kwargs[key]
        return (opt,)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPOptions"] = MZ_LLamaCPPOptions
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLamaCPPOptions"] = f"{AUTHOR_NAME} - LLamaCPPOptions"


class MZ_CustomizeInstruct:
    @classmethod
    def INPUT_TYPES(s):
        import mz_prompts
        return {
            "required": {
                "system": ("STRING", {"multiline": True, "default": mz_prompts.Long_prompt}),
                "instruct": ("STRING", {"multiline": True, "default": "Short: %text%"}),
                "start_str": ("STRING", {"default": "Long: "}),
                "end_str": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CustomizeInstruct",)
    RETURN_NAMES = ("customize_instruct",)
    FUNCTION = "create"
    CATEGORY = CATEGORY_NAME

    def create(self, **kwargs):
        return (kwargs,)


NODE_CLASS_MAPPINGS["MZ_CustomizeInstruct"] = MZ_CustomizeInstruct
NODE_DISPLAY_NAME_MAPPINGS["MZ_CustomizeInstruct"] = f"{AUTHOR_NAME} - CustomizeInstruct"


def getCommonCLIPTextEncodeInput():
    style_presets = mz_llama_cpp.get_style_presets()
    CommonCLIPTextEncodeInput = {
        "required": {
            "prompt_version": (["v1"], {"default": "v1"}),
            "style_presets": (
                style_presets, {"default": style_presets[1]}
            ),
            "text": ("STRING", {"multiline": True, }),
            "keep_device": ([False, True], {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "clip": ("CLIP", ),
            "llama_cpp_options": ("LLamaCPPOptions", ),
            "customize_instruct": ("CustomizeInstruct", ),
            # "customize_json_schema": ("STRING", ),
        }
    }

    return CommonCLIPTextEncodeInput


class MZ_LLama3CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        m_models = mz_llama3.llama3_models.copy()
        for i in range(len(m_models)):
            if mz_llama3.get_exist_model(m_models[i]) is not None:
                m_models[i] += "[downloaded]"

        importlib.reload(mz_llama_cpp)

        result = {
            "required": {
                "llama_cpp_model": (m_models, {"default": m_models[0]}),
                "download_source": (
                    ["none", "modelscope", "hf-mirror.com",],
                    {"default": "none"}
                ),
            },
            "optional": {},
        }

        common_input = getCommonCLIPTextEncodeInput()
        for key in common_input["required"]:
            result["required"][key] = common_input["required"][key]
        for key in common_input["optional"]:
            result["optional"][key] = common_input["optional"][key]

        return result

    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_llama3.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]

        return (text, conditionings)


NODE_CLASS_MAPPINGS["MZ_LLama3CLIPTextEncode"] = MZ_LLama3CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLama3CLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(LLama3)"


class MZ_Phi3CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        m_models = mz_phi3.phi3_models.copy()
        for i in range(len(m_models)):
            if mz_llama3.get_exist_model(m_models[i]) is not None:
                m_models[i] += "[downloaded]"

        importlib.reload(mz_phi3)

        result = {
            "required": {
                "llama_cpp_model": (m_models, {"default": m_models[0]}),
                "download_source": (
                    ["none", "modelscope", "hf-mirror.com",],
                    {"default": "none"}
                ),
            },
            "optional": {},
        }

        common_input = getCommonCLIPTextEncodeInput()
        for key in common_input["required"]:
            result["required"][key] = common_input["required"][key]
        for key in common_input["optional"]:
            result["optional"][key] = common_input["optional"][key]

        return result

    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_phi3.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]

        return (text, conditionings)


NODE_CLASS_MAPPINGS["MZ_Phi3CLIPTextEncode"] = MZ_Phi3CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Phi3CLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(Phi3)"


class MZ_BaseLLamaCPPCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llama_cpp)

        result = {
            "required": {
                "llama_cpp_model": ("STRING", {"default": "", "placeholder": "model_path"}),
            },
            "optional": {
            },
        }

        common_input = getCommonCLIPTextEncodeInput()
        for key in common_input["required"]:
            result["required"][key] = common_input["required"][key]
        for key in common_input["optional"]:
            result["optional"][key] = common_input["optional"][key]

        return result
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_llama_cpp.base_query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]

        return (text, conditionings)


NODE_CLASS_MAPPINGS["MZ_BaseLLamaCPPCLIPTextEncode"] = MZ_BaseLLamaCPPCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLamaCPPCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(BaseLLamaCPP)"


class MZ_LLavaImageInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llava)
        m_llava_models = mz_llava.LLava_models.copy()
        for i in range(len(m_llava_models)):
            if mz_llava.get_exist_model(m_llava_models[i]) is not None:
                m_llava_models[i] += "[downloaded]"

        m_llava_mmproj_models = mz_llava.LLava_mmproj_models.copy()
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
            "image": ("IMAGE",),
            "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
            "sd_format": (["none", "v1"], {"default": "none"}),
            "keep_device": ([False, True], {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
            "optional": {
            "clip": ("CLIP", ),
            "llama_cpp_options": ("LLamaCPPOptions", ),
            "customize_instruct": ("CustomizeInstruct", ),
        }}
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "interrogate"
    CATEGORY = CATEGORY_NAME

    def interrogate(self, **kwargs):
        importlib.reload(mz_llava)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")
        kwargs["mmproj_model"] = kwargs.get(
            "mmproj_model", "").replace("[downloaded]", "")

        kwargs["image"] = Utils.tensor2pil(kwargs["image"])

        response = mz_llava.image_interrogator(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(response)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]
        return (response, conditionings)


NODE_CLASS_MAPPINGS["MZ_LLavaImageInterrogator"] = MZ_LLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLavaImageInterrogator"] = f"{AUTHOR_NAME} - ImageInterrogator(LLava)"


class MZ_BaseLLavaImageInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "llama_cpp_model": ("STRING", {"default": ""}),
            "mmproj_model": ("STRING", {"default": ""}),
            "image": ("IMAGE",),
            "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
            "sd_format": (["none", "v1"], {"default": "none"}),
            "keep_device": ([False, True], {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
            "optional": {
            "clip": ("CLIP", ),
            "llama_cpp_options": ("LLamaCPPOptions", ),
            "customize_instruct": ("CustomizeInstruct", ),
        }}
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "interrogate"
    CATEGORY = CATEGORY_NAME

    def interrogate(self, **kwargs):
        importlib.reload(mz_llava)

        kwargs["image"] = Utils.tensor2pil(kwargs["image"])

        response = mz_llava.base_image_interrogator(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(response)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]
        return (response, conditionings)


NODE_CLASS_MAPPINGS["MZ_BaseLLavaImageInterrogator"] = MZ_BaseLLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLavaImageInterrogator"] = f"{AUTHOR_NAME} - ImageInterrogator(BaseLLava)"


class MZ_LLamaCPPInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_file": ("STRING", {"default": ""}),
                "use_system": ([True, False], {"default": True}),
                "system": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True, }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "simple_interrogator"
    CATEGORY = CATEGORY_NAME

    def simple_interrogator(self, model_file, prompt, use_system=True, system="You are a helpful assistant.", n_gpu_layers=-1):
        importlib.reload(mz_llama_cpp)
        result = mz_llama_cpp.llama_cpp_simple_interrogator(
            model_file,
            n_gpu_layers=n_gpu_layers,
            use_system=use_system,
            system=system,
            question=prompt,
        )
        return (result,)

# NODE_CLASS_MAPPINGS["MZ_LLamaCPPInterrogator"] = MZ_LLamaCPPInterrogator
# NODE_DISPLAY_NAME_MAPPINGS["MZ_LLamaCPPInterrogator"] = f"{AUTHOR_NAME} - LLamaCPP simple interrogator"


class MZ_OpenAIApiCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llama_cpp)

        result = {
            "required": {
                "base_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "placeholder": ""}),
                "model_name": ("STRING", {"default": "gpt-3.5-turbo-1106"}),
            },
            "optional": {
            },
        }

        common_input = getCommonCLIPTextEncodeInput()
        for key in common_input["required"]:
            if key not in ["seed", "keep_device"]:
                result["required"][key] = common_input["required"][key]
        for key in common_input["optional"]:
            if key != "llama_cpp_options":
                result["optional"][key] = common_input["optional"][key]

        return result
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        import mz_openaiapi

        importlib.reload(mz_openaiapi)

        text = mz_openaiapi.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True,)
            conditionings = [[cond, {"pooled_output": pooled}]]

        return (text, conditionings)


NODE_CLASS_MAPPINGS["MZ_OpenAIApiCLIPTextEncode"] = MZ_OpenAIApiCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_OpenAIApiCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(OpenAIApi)"


import mz_gen_translate
mz_gen_translate.gen_translate(NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS)
