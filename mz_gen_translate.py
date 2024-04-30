

import os
import json
import folder_paths
from pathlib import Path
COMFY_PATH = Path(folder_paths.__file__).parent


ZH_Replace_Map = {
    "llama_cpp_model": "llama.cpp模型",
    "mmproj_model": "mmproj模型",
    "resolution": "分辨率",
    "sd_format": "SD格式化",
    "image": "图像",
    "download_source": "下载源",
    "prompt_version": "提示词版本",
    "style_presets": "风格预设",
    "keep_device": "模型常驻显存",
    "llama_cpp_options": "llama.cpp可选配置",
    "Options": "可选配置",
    "LLamaCPP": "llama.cpp",
    "CLIPTextEncode": "CLIP文本编码器",
    "clip": "CLIP",
    "conditioning": "条件",
    "ImageInterrogator": "图像反推",
}


def gen_translate(NODE_DISPLAY_NAME_MAPPINGS={}, NODE_CLASS_MAPPINGS={}):
    translation_dirs = [
        os.path.join(COMFY_PATH, "custom_nodes", "AIGODLIKE-COMFYUI-TRANSLATION", "zh-CN", "Nodes"),
        os.path.join(COMFY_PATH, "custom_nodes", "AIGODLIKE-ComfyUI-Translation", "zh-CN", "Nodes"),
    ]
    translation_dir = translation_dirs[0]
    for dir in translation_dirs:
        if os.path.exists(dir):
            translation_dir = dir
            break
    translation_config = os.path.join(
        translation_dir, "ComfyUI_MinusZone.translate.json")
    if os.path.exists(translation_dir):
        if not os.path.exists(translation_config):
            with open(translation_config, "w", encoding="utf-8") as f:
                f.write("{}")

        if os.path.exists(translation_config):
            translate_config = "{}"
            with open(translation_config, "r", encoding="utf-8") as f:
                translate_config = f.read()
            nodes = json.loads(translate_config)
            for key in NODE_DISPLAY_NAME_MAPPINGS:
                if key not in nodes:
                    nodes[key] = {}

                title = NODE_DISPLAY_NAME_MAPPINGS[key]
                for k, v in ZH_Replace_Map.items():
                    title = title.replace(k, v)
                nodes[key]["title"] = title

                if key in NODE_CLASS_MAPPINGS:
                    node = NODE_CLASS_MAPPINGS[key]
                    node_INPUT_TYPES = node.INPUT_TYPES()
                    node_INPUT_TYPES_required = node_INPUT_TYPES.get(
                        "required", {})
                    nodes[key]["widgets"] = {}
                    for widget_name, _ in node_INPUT_TYPES_required.items():
                        widget_name_zh = widget_name
                        for k, v in ZH_Replace_Map.items():
                            widget_name_zh = widget_name_zh.replace(k, v)
                        nodes[key]["widgets"][widget_name] = widget_name_zh

                    node_INPUT_TYPES_optional = node_INPUT_TYPES.get(
                        "optional", {})
                    nodes[key]["inputs"] = {}
                    for widget_name, _ in node_INPUT_TYPES_optional.items():
                        widget_name_zh = widget_name
                        for k, v in ZH_Replace_Map.items():
                            widget_name_zh = widget_name_zh.replace(k, v)
                        nodes[key]["inputs"][widget_name] = widget_name_zh

                    try:
                        node_RETURN_NAMES = node.RETURN_NAMES
                        nodes[key]["outputs"] = {}
                        for widget_name in node_RETURN_NAMES:
                            widget_name_zh = widget_name
                            for k, v in ZH_Replace_Map.items():
                                widget_name_zh = widget_name_zh.replace(k, v)
                            nodes[key]["outputs"][widget_name] = widget_name_zh
                    except:
                        pass

            with open(translation_config, "w", encoding="utf-8") as f:
                f.write(json.dumps(nodes, indent=4, ensure_ascii=False))

    else:
        print("No translation dir found!")
