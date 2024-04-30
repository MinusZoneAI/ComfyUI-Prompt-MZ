![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/c5bae957-2c47-4a73-86e0-2949fcf72fd5)


# ComfyUI-Prompt-MZ
基于llama.cpp的一些和提示词相关的节点，目前包括美化提示词和类似clip-interrogator的图片反推

Use llama.cpp to assist in generating some nodes related to prompt words, including beautifying prompt words and image recognition similar to clip-interrogator

## Recent changes
* [2024-04-30] 支持自定义指令 (Support for custom instructions)
* [2024-04-30] 添加llava-v1.6-vicuna-13b (add llava-v1.6-vicuna-13b)
* [2024-04-30] 添加翻译
* [2024-04-28] 新增Phi-3-mini节点 (add Phi-3-mini node)

## Installation
1. Clone this repo into `custom_nodes` folder.
2. Restart ComfyUI.
 
## Nodes
+ CLIPTextEncode (Phi-3)

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/c4b97aeb-23c0-4cf1-a6a5-d259fdf83f6e)


+ CLIPTextEncode (LLama3)

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/40da75ab-46db-4f38-9d8e-b7f9184f77fa)


+ ImageInterrogator (LLava)

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/f397c432-c2f7-4d48-9b95-2031cfb19e8c)
  Enable parameter sd_format
  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/4d2cf65d-e8a3-4dfa-b735-9d591638028c)

+ LLamaCPPOptions

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/256483e0-c3b7-4d04-82f4-f71f7d9584c9)

+ CustomizeInstruct

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/d328ba44-2eab-4f95-bd35-585a9cdc9ec2)


+ BaseLLamaCPPCLIPTextEncode (可以手动传入模型路径/You can directly pass in the model path)
+ BaseLLavaImageInterrogator (可以手动传入模型路径/You can directly pass in the model path)

## FAQ

### LLama.dll 无法加载 (Failed to load shared library LLama.dll)
CUDA版本切换到12.1,如果你使用秋叶启动器,高级设置->环境维护->安装PyTorch->选择版本中选择CUDA 12.1的版本


### ...llama_cpp_python-0,2.63-cp310-cp310-win_and64.whl returned nonzero exit status
保持网络畅通,该上魔法上魔法,或者手动安装llama_cpp_python

