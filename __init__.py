

import json
import os
import sys
from .mz_prompt_utils import Utils
from nodes import MAX_RESOLUTION
import comfy.utils
import shutil
import comfy.samplers
import folder_paths


WEB_DIRECTORY = "./web"

AUTHOR_NAME = u"MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Prompt"

# sys.path.append(os.path.join(os.path.dirname(__file__)))

import importlib

# import mz_prompt_webserver
# mz_prompt_webserver.start_server()

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

 
from . import mz_llama_cpp
from . import mz_llava


class MZ_LLamaCPPModelConfig_ManualSelect:
    @classmethod
    def INPUT_TYPES(s):
        gguf_files = folder_paths.get_filename_list("gguf")
        return {
            "required": {
                "llama_cpp_model": (gguf_files,),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("LLamaCPPModelConfig",)
    RETURN_NAMES = ("llama_cpp_model_config",)

    FUNCTION = "create"
    CATEGORY = CATEGORY_NAME

    def create(self, **kwargs):
        return (kwargs,)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPModelConfig_ManualSelect"] = MZ_LLamaCPPModelConfig_ManualSelect
NODE_DISPLAY_NAME_MAPPINGS["MZ_LLamaCPPModelConfig_ManualSelect"] = f"{AUTHOR_NAME} - LLamaCPPModelConfigManualSelect"

class MZ_LLamaCPPCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llama_cpp)

        result = {
            "required": {
            },
            "optional": {
                "llama_cpp_model": ("LLamaCPPModelConfig",),
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
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_llama_core_nodes
        importlib.reload(mz_llama_core_nodes)

        return mz_llama_core_nodes.llama_cpp_node_encode(kwargs)


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
        from . import mz_prompts
        
        return {
            "required": {
                "system": ("STRING", {"multiline": True, "default": mz_prompts.Long_prompt}),
                "instruct": ("STRING", {"multiline": True, "default": "Short: %text%"}),
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
    from . import mz_llama_core_nodes
    style_presets = mz_llama_core_nodes.get_style_presets()
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

        return {
            "required": {
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
                "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "sd_format": (["none", "v1"], {"default": "none"}),
                "keep_device": ([False, True], {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "clip": ("CLIP", ),
                "llama_cpp_options": ("LLamaCPPOptions", ),
                "customize_instruct": ("CustomizeInstruct", ),
                "captioner_config": ("ImageCaptionerConfig", ),
            },
        }
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    OUTPUT_NODE = True
    FUNCTION = "interrogate"
    CATEGORY = CATEGORY_NAME

    def interrogate(self, **kwargs):
        importlib.reload(mz_llava)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")
        kwargs["mmproj_model"] = kwargs.get(
            "mmproj_model", "").replace("[downloaded]", "")

        if kwargs.get("image", None) is not None:
            kwargs["image"] = Utils.tensor2pil(kwargs["image"])
        else:
            kwargs["image"] = None

        text = mz_llava.image_interrogator(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_LLavaImageInterrogator"] = MZ_LLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLavaImageInterrogator"] = f"{AUTHOR_NAME} - ImageInterrogator(LLava)"


class MZ_BaseLLavaImageInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_cpp_model": ("STRING", {"default": ""}),
                "mmproj_model": ("STRING", {"default": ""}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "sd_format": (["none", "v1"], {"default": "none"}),
                "keep_device": ([False, True], {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "clip": ("CLIP", ),
                "llama_cpp_options": ("LLamaCPPOptions", ),
                "customize_instruct": ("CustomizeInstruct", ),
                "captioner_config": ("ImageCaptionerConfig", ),
            },
        }
    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    OUTPUT_NODE = True
    FUNCTION = "interrogate"
    CATEGORY = CATEGORY_NAME

    def interrogate(self, **kwargs):
        importlib.reload(mz_llava)

        if kwargs.get("image", None) is not None:
            kwargs["image"] = Utils.tensor2pil(kwargs["image"])
        else:
            kwargs["image"] = None

        text = mz_llava.base_image_interrogator(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_BaseLLavaImageInterrogator"] = MZ_BaseLLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLavaImageInterrogator"] = f"{AUTHOR_NAME} - ImageInterrogator(BaseLLava)"


class MZ_ImageCaptionerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "directory"}),
                "caption_suffix": ("STRING", {"default": ".caption"}),
                "force_update": ([False, True], {"default": False}),
            },
            "optional": {

            },
        }

    RETURN_TYPES = ("ImageCaptionerConfig",)
    RETURN_NAMES = ("captioner_config", )

    FUNCTION = "interrogate_batch"
    CATEGORY = CATEGORY_NAME

    def interrogate_batch(self, **kwargs):
        return (kwargs, )


NODE_CLASS_MAPPINGS["MZ_ImageCaptionerConfig"] = MZ_ImageCaptionerConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_ImageCaptionerConfig"] = f"{AUTHOR_NAME} - ImageCaptionerConfig"


class MZ_OpenAIApiCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llama_cpp)

        s.openai_config_path = os.path.join(
            Utils.get_models_path(),
            "openai_config.json",
        )
        default_config = {
            "base_url": "",
            "api_key": "",
            "model_name": "gpt-3.5-turbo-1106",
        }
        if os.path.exists(s.openai_config_path):
            try:
                with open(s.openai_config_path, "r", encoding="utf-8") as f:
                    default_config = json.load(f)
            except Exception as e:
                print(f"Failed to load openai_config.json: {e}")

        default_api_key = default_config.get("api_key", "")
        if default_api_key != "":
            default_api_key = default_api_key[:4] + "******"
        result = {
            "required": {
                "base_url": ("STRING", {"default": default_config.get("base_url", ""), "placeholder": ""}),
                "api_key": ("STRING", {"default": default_api_key, "placeholder": ""}),
                "model_name": ("STRING", {"default": default_config.get("model_name", ""), }),
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
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        import mz_openaiapi
        importlib.reload(mz_openaiapi)

        if kwargs.get("api_key", "").endswith("******"):
            kwargs["api_key"] = ""
            try:
                with open(self.openai_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    kwargs["api_key"] = config.get("api_key", "")
            except Exception as e:
                print(f"Failed to load openai_config.json: {e}")

        if kwargs.get("api_key", "") != "":
            with open(self.openai_config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "base_url": kwargs.get("base_url", ""),
                    "api_key": kwargs.get("api_key", ""),
                    "model_name": kwargs.get("model_name", ""),
                }, f, indent=4)
        else:
            raise ValueError("api_key is required")

        text = mz_openaiapi.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_OpenAIApiCLIPTextEncode"] = MZ_OpenAIApiCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_OpenAIApiCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(OpenAIApi)"

from . import mz_gen_translate
mz_gen_translate.gen_translate(NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS)


from .v1.init import NODE_CLASS_MAPPINGS as DEPRECATED_NODE_CLASS_MAPPINGS
from .v1.init import NODE_DISPLAY_NAME_MAPPINGS as DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(DEPRECATED_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS)
