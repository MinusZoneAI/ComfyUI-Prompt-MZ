from ..mz_prompt_utils import Utils
NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}


import importlib
from . import mz_llama3
from . import mz_phi3
from .. import mz_llama_cpp
from .. import mz_llama_core_nodes
from . import mz_deprecated
from . import mz_llava


AUTHOR_NAME = u"MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Prompt/v1"


def getCommonCLIPTextEncodeInput():
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


class MZ_LLama3CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        m_models = mz_llama3.llama3_models.copy()
        for i in range(len(m_models)):
            if mz_llama3.get_exist_model(m_models[i]) is not None:
                m_models[i] += "[downloaded]"

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
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_llama3.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_LLama3CLIPTextEncode"] = MZ_LLama3CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLama3CLIPTextEncode"] = f"{AUTHOR_NAME} - deprecated - CLIPTextEncode(LLama3)"


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
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()

        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_phi3.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_Phi3CLIPTextEncode"] = MZ_Phi3CLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Phi3CLIPTextEncode"] = f"{AUTHOR_NAME} - deprecated - CLIPTextEncode(Phi3)"


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
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        kwargs = kwargs.copy()

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")
        text = mz_deprecated.base_query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_BaseLLamaCPPCLIPTextEncode"] = MZ_BaseLLamaCPPCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLamaCPPCLIPTextEncode"] = f"{AUTHOR_NAME} - deprecated - CLIPTextEncode(BaseLLamaCPP)"


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
                "sd_format": (["none", "v1", ], {"default": "none"}),
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
        kwargs = kwargs.copy()

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

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_LLavaImageInterrogator"] = MZ_LLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LLavaImageInterrogator"] = f"{AUTHOR_NAME} - deprecated - ImageInterrogator(LLava)"


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
        kwargs = kwargs.copy()

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

        return {"ui": {"string": [Utils.to_debug_prompt(text),]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_BaseLLavaImageInterrogator"] = MZ_BaseLLavaImageInterrogator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLavaImageInterrogator"] = f"{AUTHOR_NAME} - deprecated - ImageInterrogator(BaseLLava)"
