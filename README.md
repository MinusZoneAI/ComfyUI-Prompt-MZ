![Image Description](res/1.png)

# ComfyUI-Prompt-MZ
基于llama.cpp的一些和提示词相关的节点，目前包括美化提示词和类似clip-interrogator的图片反推

Use llama.cpp to assist in generating some nodes related to prompt words, including beautifying prompt words and image recognition similar to clip-interrogator

## Recent changes
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



+ BaseLLamaCPPCLIPTextEncode (可以手动传入模型路径/You can directly pass in the model path)
+ BaseLLavaImageInterrogator (可以手动传入模型路径/You can directly pass in the model path)



