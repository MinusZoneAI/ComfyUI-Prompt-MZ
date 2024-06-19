![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/c5bae957-2c47-4a73-86e0-2949fcf72fd5)


# ComfyUI-Prompt-MZ
基于llama.cpp的一些和提示词相关的节点，目前包括美化提示词和类似clip-interrogator的图片反推

Use llama.cpp to assist in generating some nodes related to prompt words, including beautifying prompt words and image recognition similar to clip-interrogator

## Recent changes 
* [2024-06-20] 新增选择本机ollama模型的节点 (Added nodes to select local ollama models)
* [2024-06-05] 新增千问2.0预设模型 (Added Qianwen 2.0 preset model)
* [2024-06-05] 可选chat_format,图片反推后处理 (Optional chat_format, post-processing after image interrogation)
* [2024-06-04] 新增了一些预设模型 (Added some preset models)
* [2024-06-04] 新增通用节点,支持手动选择模型 (Add universal node, support manual selection of models)
* [2024-05-30] 添加ImageCaptionerConfig节点来支持批量生成提示词 (Add ImageCaptionerConfig node to support batch generation of prompt words)
* [2024-05-24] 运行后在当前节点显示生成的提示词 (Display the generated prompt words in the current node after running)
* [2024-05-24] 兼容清华智谱API (Compatible with Zhipu API)
* [2024-05-24] 使用A1111权重缩放,感谢ComfyUI_ADV_CLIP_emb (Use A1111 weight scaling, thanks to ComfyUI_ADV_CLIP_emb)
* [2024-05-13] 新增OpenAI API节点 (add OpenAI API node)
* [2024-04-30] 支持自定义指令 (Support for custom instructions)
* [2024-04-30] 添加llava-v1.6-vicuna-13b (add llava-v1.6-vicuna-13b)
* [2024-04-30] 添加翻译
* [2024-04-28] 新增Phi-3-mini节点 (add Phi-3-mini node)

## Installation
1. Clone this repo into `custom_nodes` folder.
2. Restart ComfyUI.
 
## Nodes
+ ModelConfigManualSelect (Ollama)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/2009e330-0f1f-4f28-9b4c-8446d3cdc519)


+ CLIPTextEncode (LLamaCPP Universal)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/1f66ce10-920f-4ada-9287-f86a51782bff)


+ ModelConfigManualSelect(LLamaCPP)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/42473248-8902-43d7-a08b-37bb3d20b4aa)

+ ModelConfigDownloaderSelect(LLamaCPP)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/6a2f561b-deb0-43d3-900f-c9d6b23d0ea4)



+ CLIPTextEncode (ImageInterrogator)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/e76eb5dc-1c6c-4a59-8197-8bd7b56c3889)

+ ModelConfigManualSelect(ImageInterrogator)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/71a48734-e3f3-4ced-a8d7-cd334340efdb)


+ ModelConfigDownloaderSelect(ImageInterrogator)
  
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/bfec7696-1f86-4fe5-9dc3-807b39366524)



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

+ ImageCaptionerConfig
![image](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ/assets/5035199/147941a2-cb5f-418f-acd9-8e17ffaf044a)


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




## Credits
+ [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
+ [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
+ [https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)

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
