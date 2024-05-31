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


AUTHOR_NAME = u"MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Prompt/deprecated"


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

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


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
        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_phi3.query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


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
        importlib.reload(mz_llama3)

        kwargs["llama_cpp_model"] = kwargs.get(
            "llama_cpp_model", "").replace("[downloaded]", "")

        text = mz_llama_cpp.base_query_beautify_prompt_text(kwargs)
        conditionings = None
        clip = kwargs.get("clip", None)
        if clip is not None:
            conditionings = Utils.a1111_clip_text_encode(clip, text, )

        return {"ui": {"string": [text,]}, "result": (text, conditionings)}


NODE_CLASS_MAPPINGS["MZ_BaseLLamaCPPCLIPTextEncode"] = MZ_BaseLLamaCPPCLIPTextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_BaseLLamaCPPCLIPTextEncode"] = f"{AUTHOR_NAME} - deprecated - CLIPTextEncode(BaseLLamaCPP)"
