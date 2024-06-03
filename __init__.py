

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

from . import mz_prompt_webserver
mz_prompt_webserver.start_server()

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}


from . import mz_llama_cpp


def getCommonCLIPTextEncodeInput():
    from . import mz_llama_core_nodes
    style_presets = mz_llama_core_nodes.get_style_presets()
    CommonCLIPTextEncodeInput = {
        "required": {
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
        }
    }

    return CommonCLIPTextEncodeInput


class MZ_LLamaCPPModelConfig_ManualSelect:
    @classmethod
    def INPUT_TYPES(s):
        gguf_files = Utils.get_gguf_files()
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
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()

        llama_cpp_model = kwargs.get("llama_cpp_model", "")
        if llama_cpp_model != "":
            llama_cpp_model = os.path.join(
                Utils.get_gguf_models_path(), llama_cpp_model)

        return ({
            "type": "ManualSelect",
            "model_path": llama_cpp_model,
        },)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPModelConfig_ManualSelect"] = MZ_LLamaCPPModelConfig_ManualSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLamaCPPModelConfig_ManualSelect"] = f"{AUTHOR_NAME} - ModelConfigManualSelect(LLamaCPP)"


class MZ_LLamaCPPModelConfig_DownloaderSelect:
    @classmethod
    def INPUT_TYPES(s):
        optional_models = Utils.get_model_zoo(tags_filter="llama")
        model_names = [
            model["model"] for model in optional_models
        ]
        return {
            "required": {
                "model_name": (model_names,),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("LLamaCPPModelConfig",)
    RETURN_NAMES = ("llama_cpp_model_config",)

    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()

        model_name = kwargs.get("model_name", "")
        return ({
            "type": "DownloaderSelect",
            "model_name": model_name,
        },)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPModelConfig_DownloaderSelect"] = MZ_LLamaCPPModelConfig_DownloaderSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLamaCPPModelConfig_DownloaderSelect"] = f"{AUTHOR_NAME} - ModelConfigDownloaderSelect(LLamaCPP)"


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

    DESCRIPTION = """
llama_cpp_model不设置时，将使用默认模型: Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
"""

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_llama_core_nodes
        importlib.reload(mz_llama_core_nodes)

        return mz_llama_core_nodes.llama_cpp_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPCLIPTextEncode"] = MZ_LLamaCPPCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLamaCPPCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(LLamaCPP Universal)"


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
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()
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
                "instruct": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("CustomizeInstruct",)
    RETURN_NAMES = ("customize_instruct",)
    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()

        return (kwargs,)


NODE_CLASS_MAPPINGS["MZ_CustomizeInstruct"] = MZ_CustomizeInstruct
NODE_DISPLAY_NAME_MAPPINGS["MZ_CustomizeInstruct"] = f"{AUTHOR_NAME} - CustomizeInstruct"


class MZ_ImageCaptionerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "directory"}),
                "caption_suffix": ("STRING", {"default": ".caption"}),
                "force_update": ([False, True], {"default": False}),
                "retry_keyword": ("STRING", {"default": "not,\",error"}),
            },
            "optional": {

            },
        }

    RETURN_TYPES = ("ImageCaptionerConfig",)
    RETURN_NAMES = ("captioner_config", )

    FUNCTION = "interrogate_batch"
    CATEGORY = f"{CATEGORY_NAME}/others"

    def interrogate_batch(self, **kwargs):
        kwargs = kwargs.copy()

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
        kwargs = kwargs.copy()

        from . import mz_openaiapi
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

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_OpenAIApiCLIPTextEncode"] = MZ_OpenAIApiCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_OpenAIApiCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(OpenAIApi)"


class MZ_ImageInterrogatorCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": ("INT", {"default": 512, "min": 128, "max": 0xffffffffffffffff}),
                "keep_device": ([False, True], {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image_interrogator_model": ("ImageInterrogatorModelConfig", ),
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
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_llama_core_nodes
        importlib.reload(mz_llama_core_nodes)

        return mz_llama_core_nodes.image_interrogator_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_ImageInterrogatorCLIPTextEncode"] = MZ_ImageInterrogatorCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ImageInterrogatorCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(ImageInterrogator)"


class MZ_ImageInterrogatorModelConfig_ManualSelect:
    @classmethod
    def INPUT_TYPES(s):
        gguf_files = Utils.get_gguf_files()
        return {
            "required": {
                "llama_cpp_model": (gguf_files,),
                "mmproj_model": (["auto"] + gguf_files,),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("ImageInterrogatorModelConfig",)
    RETURN_NAMES = ("image_interrogator_model",)

    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()

        llama_cpp_model = kwargs.get("llama_cpp_model", "")
        if llama_cpp_model != "":
            llama_cpp_model = os.path.join(
                Utils.get_gguf_models_path(), llama_cpp_model)

        mmproj_model = kwargs.get("mmproj_model", "")
        if mmproj_model != "":
            mmproj_model = os.path.join(
                Utils.get_gguf_models_path(), mmproj_model)

        return ({
            "type": "ManualSelect",
            "model_path": llama_cpp_model,
            "mmproj_model": mmproj_model,
        },)


NODE_CLASS_MAPPINGS["MZ_ImageInterrogatorModelConfig_ManualSelect"] = MZ_ImageInterrogatorModelConfig_ManualSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ImageInterrogatorModelConfig_ManualSelect"] = f"{AUTHOR_NAME} - ModelConfigManualSelect(ImageInterrogator)"


class MZ_ImageInterrogatorModelConfig_DownloaderSelect:
    @classmethod
    def INPUT_TYPES(s):
        optional_models = Utils.get_model_zoo(tags_filter="llava")
        model_names = [
            model["model"] for model in optional_models
        ]

        optional_models = Utils.get_model_zoo(tags_filter="mmproj")
        mmproj_model_names = [
            model["model"] for model in optional_models
        ]

        return {
            "required": {
                "model_name": (model_names,),
                "mmproj_model_name": (mmproj_model_names,),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("ImageInterrogatorModelConfig",)
    RETURN_NAMES = ("image_interrogator_model",)

    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_NAME}/others"

    def create(self, **kwargs):
        kwargs = kwargs.copy()
        model_name = kwargs.get("model_name")
        mmproj_model_name = kwargs.get("mmproj_model_name", "auto")
        return ({
            "type": "DownloaderSelect",
            "model_name": model_name,
            "mmproj_model_name": mmproj_model_name,
        },)


NODE_CLASS_MAPPINGS["MZ_ImageInterrogatorModelConfig_DownloaderSelect"] = MZ_ImageInterrogatorModelConfig_DownloaderSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ImageInterrogatorModelConfig_DownloaderSelect"] = f"{AUTHOR_NAME} - ModelConfigDownloaderSelect(ImageInterrogator)"


from . import mz_gen_translate
mz_gen_translate.gen_translate(NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS)


from .v1.init import NODE_CLASS_MAPPINGS as DEPRECATED_NODE_CLASS_MAPPINGS
from .v1.init import NODE_DISPLAY_NAME_MAPPINGS as DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(DEPRECATED_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS)
