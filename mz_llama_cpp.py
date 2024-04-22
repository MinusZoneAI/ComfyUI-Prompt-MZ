import os 
import shutil
import subprocess
import sys
import torch
try:
    import mz_utils
except ImportError:
    pass
  
def check_llama_cpp_requirements(): 
    last_version = "0.2.63"
    try:
        from llama_cpp import Llama
    except ImportError:
        py_version = ""
        if sys.version_info.major == 3:
            if sys.version_info.minor == 10:
                py_version = "310"
            elif sys.version_info.minor == 11:
                py_version = "311"
            elif sys.version_info.minor == 12:
                py_version = "312"

        if py_version == "":
            raise ValueError(f"Please upgrade python to version 3.10 or above. (找不到对应的python版本) 当前版本:{sys.version_info.major}.{sys.version_info.minor}")

        cuda_version = ""
        if torch.cuda.is_available():
            cuda_version = "cu" + torch.version.cuda.replace(".", "") 
            if cuda_version not in ["cu121", "cu122", "cu123"]:
                cuda_version = "cu121"
        else:
            cuda_version = "cpu"

        # https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.63-cu123/llama_cpp_python-0.2.63-cp310-cp310-linux_x86_64.whl

        system_name = "linux_x86_64"
        if sys.platform == "linux":
            if sys.maxsize > 2**32:
                system_name = "linux_x86_64"
            else:
                system_name = "linux_i686"
        elif sys.platform == "darwin":
            # 请手动前往https://github.com/abetlen/llama-cpp-python/releases 下载对应的whl文件后 使用pip install {whl文件路径}安装
            raise ValueError("Please download the corresponding whl file from https://github.com/abetlen/llama-cpp-python/releases and install it using pip install {whl file path} (请手动前往https://github.com/abetlen/llama-cpp-python/releases 下载对应的whl文件后 使用pip install {whl文件路径}安装)")
        elif sys.platform == "win32":
            system_name = "win_amd64"
        else:
            raise ValueError(f"Unsupported platform. (不支持的平台) {sys.platform} (请手动前往https://github.com/abetlen/llama-cpp-python/releases 下载对应的whl文件后 使用pip install 'whl文件路径' 安装)")

        wheel_name = f"llama_cpp_python-{last_version}-cp{py_version}-cp{py_version}-{system_name}.whl"
        if cuda_version == "cpu":
            wheel_url = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{last_version}/{wheel_name}"
        else:


            wheel_url = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{last_version}-{cuda_version}/{wheel_name}"

        print(f"pip install {wheel_url}")
        ret = subprocess.run([
            sys.executable, "-m",
            "pip", "install", wheel_url], check=True)
        
        if ret.returncode != 0:
            raise ValueError("Failed to install llama_cpp. (安装llama_cpp失败)")
        else: 
            print("llama_cpp installed successfully. (llama_cpp安装成功)")
        



def llama_cpp_messages(model_file, n_gpu_layers, chat_handler=None, messages=[], options={}):
    print(f"Find local model file: {model_file}")

    check_llama_cpp_requirements()

    from llama_cpp import Llama

    model_and_opt = mz_utils.Utils.cache_get(f"llama_cpp_model_and_opt_{model_file}")
    
    init_opts = ["n_ctx"]
    # compared
    is_opts_changed = False

    if model_and_opt is not None:
        for opt in init_opts:
            if model_and_opt.get("options").get(opt) != options.get(opt):
                is_opts_changed = True
                break

    if model_and_opt is None or is_opts_changed:
        print("llama_cpp: loading model...")
        verbose = False
        if os.environ.get("MZ_DEV", None) is not None:
            verbose = True

        model = Llama(
            model_path=model_file, 
            n_gpu_layers=n_gpu_layers,
            n_ctx=options.get("n_ctx", 1024),
            logits_all=options.get("logits_all", False),
            chat_handler=chat_handler,
            chat_format=options.get("chat_format", None),
            verbose=verbose,
        )   
        model_and_opt = {
            "model": model,
            "options": options,
        }
        mz_utils.Utils.cache_set(f"llama_cpp_model_and_opt_{model_file}", model_and_opt)

    
    model = model_and_opt.get("model")


    response_format = options.get("response_format", None)
    mz_utils.Utils.print_log(f"======================================================LLAMA_CPP======================================================")
    # mz_utils.Utils.print_log("llama_cpp messages:", messages) 
    mz_utils.Utils.print_log("llama_cpp response_format:", response_format) 
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=options.get("max_tokens", None),
        response_format=response_format,
        temperature=options.get("temperature", 0.8),
    )
    mz_utils.Utils.print_log(f"LLAMA_CPP: \n{output}")
    choices = output.get("choices", [])
    # mz_utils.Utils.print_log(f"LLAMA_CPP choices: \n{choices}")
    if len(choices) == 0:
        return ""
    
    result = choices[0].get("message", {}).get("content", "")

    return result

def get_schema_base_type(t):
    return {
        "type": t,
    }

def get_schema_obj(keys_type={}, required=[]):
    item = {}
    for key, value in keys_type.items():
        item[key] = value
    return {
        "type": "object",
        "properties": item,
        "required": required
    }

def get_schema_array(item_type="string"):
    if type(item_type) == str:
        item_type = get_schema_base_type(item_type)
    return {
        "type": "array",
        "items": item_type,
    }



def llama_cpp_simple_interrogator_to_json(model_file, n_gpu_layers, use_system=True, system=None, question="", schema={}, max_tokens=1024, temperature=0.8):

    if system is None:
        system = ""
        messages = [
            {
                "role": "user",
                "content": question
            },
        ]
    elif use_system:
        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": question
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": f"{system}\nIf you understand what I am saying, please reply 'OK' and do not reply with unnecessary content."
            },
            {
                "role": "assistant",
                "content": "OK"
            },
            {
                "role": "user",
                "content": question
            },
        ]

    response_format = {
        "type": "json_object",
        "schema": schema,
    } 


    return llama_cpp_messages(model_file, n_gpu_layers, None, messages, options={
        "response_format": response_format,
        "chat_format":"chatml",
        "temperature": temperature,
        "max_tokens": max_tokens,  
    })
    
def llama_cpp_simple_interrogator(model_file, n_gpu_layers, use_system=True, system=None, question=""):
    if system is None:
        system = ""
        messages = [
            {
                "role": "user",
                "content": question
            },
        ]
    elif use_system:
        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": question
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": f"{system}\nIf you understand what I am saying, please reply 'OK' and do not reply with unnecessary content."
            },
            {
                "role": "assistant",
                "content": "OK"
            },
            {
                "role": "user",
                "content": question
            },
        ]
    return llama_cpp_messages(model_file, n_gpu_layers, None, messages)


def llava_cpp_messages(model_file, n_gpu_layers, chat_handler, messages, options={}):
    options["logits_all"] = True
    options["n_ctx"] = max(2048, options.get("n_ctx", 2048)) 
    return llama_cpp_messages(model_file, n_gpu_layers, chat_handler, messages, options)

 

def llava_cpp_simple_interrogator(model_file, mmproj_file, n_gpu_layers, system="You are an assistant who perfectly describes images.", question="Describe this image in detail please.", image=None):
    check_llama_cpp_requirements()

    content = []
    if image is not None:
        data_uri = mz_utils.Utils.pil_image_to_base64(image)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
        
    content.append({"type" : "text", "text": question})
        
    check_llama_cpp_requirements()
    from llama_cpp.llama_chat_format import Llava15ChatHandler  
    if mmproj_file is not None:
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_file)

    return llava_cpp_messages(model_file, n_gpu_layers, chat_handler, [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": content,
        }, 
    ])





if __name__ == "__main__":
    check_llama_cpp_requirements()
    llama_cpp_simple_interrogator(
        "D:\下载\gemma-1.1-2b-it-IQ3_M.gguf", 
        use_system=False,
        system="""
Stable Diffusion is an AI art generation model similar to DALLE-2.
Below is a list of prompts that can be used to generate images with Stable Diffusion:
- portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski
- pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson
- ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image
- red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski
- a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt
- athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration
- closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo
- ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha
I want you to write me a list of detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more.
Please generate the long prompt version of the short one according to the given examples. Long prompt version should consist of 3 to 5 sentences. Long prompt version must sepcify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe any atmosphere!!!
""",
        question="IDEA: 一匹马",
    )
