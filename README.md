![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/c5bae957-2c47-4a73-86e0-2949fcf72fd5)


# ComfyUI-Prompt-MZ
基于llama.cpp的一些和提示词相关的节点，目前包括美化提示词和类似clip-interrogator的图片反推

Use llama.cpp to assist in generating some nodes related to prompt words, including beautifying prompt words and image recognition similar to clip-interrogator

## Recent changes
* [2024-05-13] 新增OpenAI API节点 (add OpenAI API node)
* [2024-04-30] 支持自定义指令 (Support for custom instructions)
* [2024-04-30] 添加llava-v1.6-vicuna-13b (add llava-v1.6-vicuna-13b)
* [2024-04-30] 添加翻译
* [2024-04-28] 新增Phi-3-mini节点 (add Phi-3-mini node)

## Installation
1. Clone this repo into `custom_nodes` folder.
2. Restart ComfyUI.
 
## Nodes
+ CLIPTextEncode (OpenAI API)

  ![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/14e9a96a-ec1b-481d-8f5a-43cd752ad01b)



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

### moudle 'llama_cpp' has no attribute 'LLAMA_SPLIT_MODE_LAYER'
升级llama_cpp_python的版本到最新版本,前往 https://github.com/abetlen/llama-cpp-python/releases 下载安装

### LLama.dll 无法加载 (Failed to load shared library LLama.dll)
CUDA版本切换到12.1,如果你使用秋叶启动器,高级设置->环境维护->安装PyTorch->选择版本中选择CUDA 12.1的版本


### ...llama_cpp_python-0,2.63-cp310-cp310-win_and64.whl returned nonzero exit status
保持网络畅通,该上魔法上魔法,或者手动安装llama_cpp_python

## Star History

<a href="https://star-history.com/#MinusZoneAI/ComfyUI-Prompt-MZ&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Prompt-MZ&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Prompt-MZ&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Prompt-MZ&type=Date" />
 </picture>
</a>

## Contact
- 绿泡泡: minrszone
- Bilibili: [minus_zone](https://space.bilibili.com/5950992)
- 小红书: [MinusZoneAI](https://www.xiaohongshu.com/user/profile/5f072e990000000001005472)
- 爱发电: [MinusZoneAI](https://afdian.net/@MinusZoneAI)
