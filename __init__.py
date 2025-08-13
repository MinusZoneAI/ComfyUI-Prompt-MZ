

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
# mz_prompt_webserver.start_server()

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


class MZ_OllamaModelConfig_ManualSelect:
    @classmethod
    def INPUT_TYPES(s):
        search_dirs = [
            os.path.join(os.path.expanduser('~'), ".ollama", "models"),
            os.path.join(os.environ.get("APPDATA", ""), ".ollama", "models"),
        ]

        ollama_models_dir = None
        for dir in search_dirs:
            if os.path.exists(dir):
                ollama_models_dir = dir
                break

        ollamas = []
        if ollama_models_dir is not None:
            manifests_dir = os.path.join(ollama_models_dir, "manifests")
            for root, dirs, files in os.walk(manifests_dir):
                for file in files:
                    ollamas.append(os.path.join(root, file))

        chat_format = mz_llama_cpp.get_llama_cpp_chat_handlers()
        return {
            "required": {
                "ollama": (ollamas,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
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

        ollama = kwargs.get("ollama", "")
        ollama_cpp_model = None
        if os.path.exists(ollama):
            # {"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:887433b89a901c156f7e6944442f3c9e57f3c55d6ed52042cbb7303aea994290","size":483},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12","size":1678447520},{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:097a36493f718248845233af1d3fefe7a303f864fae13bc31a3a9704229378ca","size":8433},{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:109037bec39c0becc8221222ae23557559bc594290945a2c4221ab4f303b8871","size":136},{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:22a838ceb7fb22755a3b0ae9b4eadde629d19be1f651f73efb8c6b4e2cd0eea0","size":84}]}
            with open(ollama, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "layers" in data:
                    for layer in data["layers"]:
                        if "mediaType" in layer and layer["mediaType"] == "application/vnd.ollama.image.model":
                            ollama_cpp_model = layer["digest"]
                            break

        if ollama_cpp_model is None:
            raise ValueError("Invalid ollama file")

        if ollama_cpp_model.startswith("sha256:"):
            ollama_cpp_model = ollama_cpp_model[7:]
        # ollama = C:\Users\admin\.ollama\models\manifests\registry.ollama.ai\library\gemma\2b
        models_dir = ollama[:ollama.rfind("manifests")]
        ollama_cpp_model = os.path.join(
            models_dir, "blobs", f"sha256-{ollama_cpp_model}")

        if not os.path.exists(ollama_cpp_model):
            raise ValueError(f"Model not found at: {ollama_cpp_model}")

        llama_cpp_model = ollama_cpp_model

        chat_format = kwargs.get("chat_format", "auto")
        if chat_format == "auto":
            chat_format = None
        return ({
            "type": "ManualSelect",
            "model_path": llama_cpp_model,
            "chat_format": chat_format,
        },)


NODE_CLASS_MAPPINGS["MZ_OllamaModelConfig_ManualSelect"] = MZ_OllamaModelConfig_ManualSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_OllamaModelConfig_ManualSelect"] = f"{AUTHOR_NAME} - ModelConfigManualSelect(OllamaFile)"


class MZ_LLamaCPPModelConfig_ManualSelect:
    @ classmethod
    def INPUT_TYPES(s):
        gguf_files = Utils.get_gguf_files()

        chat_format = mz_llama_cpp.get_llama_cpp_chat_handlers()
        return {
            "required": {
                "llama_cpp_model": (gguf_files,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
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

        chat_format = kwargs.get("chat_format", "auto")
        if chat_format == "auto":
            chat_format = None
        return ({
            "type": "ManualSelect",
            "model_path": llama_cpp_model,
            "chat_format": chat_format,
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
        chat_format = mz_llama_cpp.get_llama_cpp_chat_handlers()
        return {
            "required": {
                "model_name": (model_names,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
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
        chat_format = kwargs.get("chat_format", "auto")
        if chat_format == "auto":
            chat_format = None
        return ({
            "type": "DownloaderSelect",
            "model_name": model_name,
            "chat_format": chat_format,
        },)


NODE_CLASS_MAPPINGS["MZ_LLamaCPPModelConfig_DownloaderSelect"] = MZ_LLamaCPPModelConfig_DownloaderSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLamaCPPModelConfig_DownloaderSelect"] = f"{AUTHOR_NAME} - ModelConfigDownloaderSelect(LLamaCPP)"


class MZ_LLamaCPPCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_llama_cpp)
        from . import mz_llama_core_nodes
        style_presets = mz_llama_core_nodes.get_style_presets()

        return {
            "required": {
                "style_presets": (style_presets, {"default": style_presets[1]}),
                "text": ("STRING", {"multiline": True}),
                "translate": ([False, True], {"default": False}),
                "keep_device": ([False, True], {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "clip": ("CLIP",),
                "llama_cpp_model": ("LLamaCPPModelConfig",),
                "llama_cpp_options": ("LLamaCPPOptions",),
                "customize_instruct": ("CustomizeInstruct",),
            },
        }

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

        # if opt.get("chat_format", None) == "auto":
        #     opt["chat_format"] = None
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
                "prompt_fixed_beginning": ("STRING", {"default": "", }),
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
                "post_processing": ([False, True], {"default": True}),
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
        chat_format = mz_llama_cpp.get_llama_cpp_chat_handlers()
        return {
            "required": {
                "llama_cpp_model": (gguf_files,),
                "mmproj_model": (["auto"] + gguf_files,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
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

        chat_format = kwargs.get("chat_format", "auto")
        if chat_format == "auto":
            chat_format = None
        return ({
            "type": "ManualSelect",
            "model_path": llama_cpp_model,
            "mmproj_model": mmproj_model,
            "chat_format": chat_format,
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

        chat_format = mz_llama_cpp.get_llama_cpp_chat_handlers()
        return {
            "required": {
                "model_name": (model_names,),
                "mmproj_model_name": (["auto"] + mmproj_model_names,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
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
        chat_format = kwargs.get("chat_format", "auto")
        if chat_format == "auto":
            chat_format = None
        return ({
            "type": "DownloaderSelect",
            "model_name": model_name,
            "mmproj_model_name": mmproj_model_name,
            "chat_format": chat_format,
        },)


NODE_CLASS_MAPPINGS["MZ_ImageInterrogatorModelConfig_DownloaderSelect"] = MZ_ImageInterrogatorModelConfig_DownloaderSelect
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ImageInterrogatorModelConfig_DownloaderSelect"] = f"{AUTHOR_NAME} - ModelConfigDownloaderSelect(ImageInterrogator)"


class MZ_Florence2CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "Florence-2-large-ft",
                    "Florence-2-large",
                ],),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 0xffffffffffffffff}),
                "keep_device": ([False, True], {"default": False}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "clip": ("CLIP", ),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_transformers
        importlib.reload(mz_transformers)

        return mz_transformers.florence2_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_Florence2CLIPTextEncode"] = MZ_Florence2CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Florence2CLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(Florence-2)"


class MZ_Florence2Captioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "Florence-2-large-ft",
                    "Florence-2-large",
                ],),
                "directory": ("STRING", {"default": "", "placeholder": "directory"}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "caption_suffix": ("STRING", {"default": ".caption"}),
                "force_update": ([False, True], {"default": False}),
                "prompt_fixed_beginning": ("STRING", {"default": "", }),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_transformers
        importlib.reload(mz_transformers)

        kwargs["captioner_config"] = {
            "directory": kwargs["directory"],
            "resolution": kwargs["resolution"],
            "batch_size": kwargs["batch_size"],
            "caption_suffix": kwargs["caption_suffix"],
            "force_update": kwargs["force_update"],
            "prompt_fixed_beginning": kwargs["prompt_fixed_beginning"],
        }

        return mz_transformers.florence2_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_Florence2Captioner"] = MZ_Florence2Captioner
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Florence2Captioner"] = f"{AUTHOR_NAME} - Captioner(Florence-2)"


class MZ_PaliGemmaCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "paligemma-sd3-long-captioner-v2",
                    "paligemma-sd3-long-captioner",
                ],),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 0xffffffffffffffff}),
                "keep_device": ([False, True], {"default": False}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "clip": ("CLIP", ),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_transformers
        importlib.reload(mz_transformers)

        return mz_transformers.paligemma_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_PaliGemmaCLIPTextEncode"] = MZ_PaliGemmaCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_PaliGemmaCLIPTextEncode"] = f"{AUTHOR_NAME} - CLIPTextEncode(PaliGemma)"


class MZ_PaliGemmaCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "paligemma-sd3-long-captioner-v2",
                    "paligemma-sd3-long-captioner",
                ],),
                "directory": ("STRING", {"default": "", "placeholder": "directory"}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 0xffffffffffffffff}),
                "caption_suffix": ("STRING", {"default": ".caption"}),
                "force_update": ([False, True], {"default": False}),
                "prompt_fixed_beginning": ("STRING", {"default": "", }),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()
        from . import mz_transformers
        importlib.reload(mz_transformers)
        kwargs["captioner_config"] = {
            "directory": kwargs["directory"],
            "resolution": kwargs["resolution"],
            "caption_suffix": kwargs["caption_suffix"],
            "force_update": kwargs["force_update"],
            "prompt_fixed_beginning": kwargs["prompt_fixed_beginning"],
        }
        return mz_transformers.paligemma_node_encode(kwargs)


NODE_CLASS_MAPPINGS["MZ_PaliGemmaCaptioner"] = MZ_PaliGemmaCaptioner
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_PaliGemmaCaptioner"] = f"{AUTHOR_NAME} - Captioner(PaliGemma)"

try:
    from . import mz_gen_translate
    mz_gen_translate.gen_translate(
        NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS)
except Exception as e:
    print(f"Failed to generate translation: {e}")


from .v1.init import NODE_CLASS_MAPPINGS as DEPRECATED_NODE_CLASS_MAPPINGS
from .v1.init import NODE_DISPLAY_NAME_MAPPINGS as DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(DEPRECATED_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DEPRECATED_NODE_DISPLAY_NAME_MAPPINGS)
