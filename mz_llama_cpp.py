import importlib
import json
import os
import shutil
import subprocess
import sys
import torch
try:
    from . import mz_prompt_utils
    from . import mz_prompt_webserver
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
            raise ValueError(
                f"Please upgrade python to version 3.10 or above. (找不到对应的python版本) 当前版本:{sys.version_info.major}.{sys.version_info.minor}")

        cuda_version = ""
        if torch.cuda.is_available():
            cuda_version = "cu" + torch.version.cuda.replace(".", "")
            if cuda_version not in ["cu121", "cu122", "cu123"]:
                cuda_version = "cu121"
                print(
                    f"Warning: The current version of cuda is not supported. (警告: 当前cuda版本不支持) {torch.version.cuda} (默认使用cu121)")
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
            raise ValueError(
                "Please download the corresponding whl file from https://github.com/abetlen/llama-cpp-python/releases and install it using pip install {whl file path} (请手动前往https://github.com/abetlen/llama-cpp-python/releases 下载对应的whl文件后 使用pip install {whl文件路径}安装)")
        elif sys.platform == "win32":
            system_name = "win_amd64"
        else:
            raise ValueError(
                f"Unsupported platform. (不支持的平台) {sys.platform} (请手动前往https://github.com/abetlen/llama-cpp-python/releases 下载对应的whl文件后 使用pip install 'whl文件路径' 安装)")

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


def LlamaCppOptions():
    return {
        "n_ctx": 2048,
        "n_batch": 2048,
        "n_threads": 0,
        "n_threads_batch": 0,
        "split_mode": ["LLAMA_SPLIT_MODE_NONE", "LLAMA_SPLIT_MODE_LAYER", "LLAMA_SPLIT_MODE_ROW",],
        "main_gpu": 0,
        "n_gpu_layers": -1,
        "max_tokens": 4096,
        "temperature": 1.6,
        "top_p": 0.95,
        "min_p": 0.05,
        "typical_p": 1.0,
        "stop": "",
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.1,
        "top_k": 50,
        "tfs_z": 1.0,
        "mirostat_mode": ["none", "mirostat", "mirostat_v2"],
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
    }


def freed_gpu_memory(model_file):
    check_llama_cpp_requirements()

    model_and_opt = mz_prompt_utils.Utils.cache_get(
        f"llama_cpp_model_and_opt_{model_file}")

    if model_and_opt is None:
        return 0

    model = model_and_opt.get("model")

    del model
    torch.cuda.empty_cache()

    mz_prompt_utils.Utils.cache_set(
        f"llama_cpp_model_and_opt_{model_file}", None)


def llama_cpp_messages(model_file, chat_handler=None, messages=[], options={}):
    if options is None:
        options = {}
    print(f"Find local model file: {model_file}")
    init_opts = ["n_ctx", "logits_all", "chat_format", "n_gpu_layers"]

    check_llama_cpp_requirements()

    from llama_cpp import Llama
    import llama_cpp

    model_and_opt = mz_prompt_utils.Utils.cache_get(
        f"llama_cpp_model_and_opt_{model_file}")

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

        split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_LAYER
        if options.get("split_mode", "LLAMA_SPLIT_MODE_LAYER") == "LLAMA_SPLIT_MODE_ROW":
            split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_ROW
        elif options.get("split_mode", "LLAMA_SPLIT_MODE_LAYER") == "LLAMA_SPLIT_MODE_NONE":
            split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_NONE

        model = Llama(
            model_path=model_file,
            n_gpu_layers=options.get("n_gpu_layers", -1),
            n_ctx=options.get("n_ctx", 2048),
            n_batch=options.get("n_batch", 2048),
            n_threads=options.get("n_threads", 0) if options.get(
                "n_threads", 0) > 0 else None,
            n_threads_batch=options.get("n_threads_batch", 0) if options.get(
                "n_threads_batch", 0) > 0 else None,
            main_gpu=options.get("main_gpu", 0),
            split_mode=split_mode_int,
            logits_all=options.get("logits_all", False),
            chat_handler=chat_handler,
            chat_format=options.get("chat_format", None),
            seed=options.get("seed", -1),
            verbose=verbose,
        )
        model_and_opt = {
            "model": model,
            "options": options,
        }
        mz_prompt_utils.Utils.cache_set(
            f"llama_cpp_model_and_opt_{model_file}", model_and_opt)

    model = model_and_opt.get("model")

    response_format = options.get("response_format", None)
    mz_prompt_utils.Utils.print_log(
        f"======================================================LLAMA_CPP======================================================")
    # mz_utils.Utils.print_log("llama_cpp messages:", messages)
    mz_prompt_utils.Utils.print_log(
        "llama_cpp response_format:", response_format)

    stop = options.get("stop", "")
    if stop == "":
        stop = []
    else:
        # 所有转译序列
        escape_sequence = {
            "\\n": "\n",
            "\\t": "\t",
            "\\r": "\r",
            "\\b": "\b",
            "\\f": "\f",
        }
        for key, value in escape_sequence.items():
            stop = stop.replace(key, value)
        stop = stop.split(",")

    mirostat_mode = 0
    if options.get("mirostat_mode", "none") == "mirostat":
        mirostat_mode = 1
    elif options.get("mirostat_mode", "none") == "mirostat_v2":
        mirostat_mode = 2
    output = model.create_chat_completion(
        messages=messages,
        response_format=response_format,
        max_tokens=options.get("max_tokens", 4096),
        temperature=options.get("temperature", 1.6),
        top_p=options.get("top_p", 0.95),
        min_p=options.get("min_p", 0.05),
        typical_p=options.get("typical_p", 1.0),
        stop=stop,
        frequency_penalty=options.get("frequency_penalty", 0.0),
        presence_penalty=options.get("presence_penalty", 0.0),
        repeat_penalty=options.get("repeat_penalty", 1.1),
        top_k=options.get("top_k", 50),
        tfs_z=options.get("tfs_z", 1.0),
        mirostat_mode=mirostat_mode,
        mirostat_tau=options.get("mirostat_tau", 5.0),
        mirostat_eta=options.get("mirostat_eta", 0.1),
    )
    mz_prompt_utils.Utils.print_log(f"LLAMA_CPP: \n{output}")
    choices = output.get("choices", [])
    # mz_utils.Utils.print_log(f"LLAMA_CPP choices: \n{choices}")
    if len(choices) == 0:
        return ""

    result = choices[0].get("message", {}).get("content", "")
    return result



def llama_cpp_simple_interrogator_to_json(model_file, use_system=True, system=None, question="", schema={}, options={}):
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

    options["response_format"] = response_format
    options["chat_format"] = "chatml"

    result = llama_cpp_messages(model_file, None, messages, options=options)
    result = result.replace("\n", " ")
    return result


def llama_cpp_simple_interrogator(model_file, use_system=True, system=None, question="", options={}):
    if options is None:
        options = {}
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
    return llama_cpp_messages(model_file, None, messages, options=options)


def llava_cpp_messages(model_file, chat_handler, messages, options={}):
    if options is None:
        options = {}
    options["logits_all"] = True
    options["n_ctx"] = max(4096, options.get("n_ctx", 4096))
    return llama_cpp_messages(model_file, chat_handler, messages, options)


def llava_cpp_simple_interrogator(
        model_file, mmproj_file, system="You are an assistant who perfectly describes images.", question="Describe this image in detail please.",
        image=None, options={}):
    if options is None:
        options = {}

    check_llama_cpp_requirements()

    content = []
    if image is not None:
        data_uri = mz_prompt_utils.Utils.pil_image_to_base64(image)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    content.append({"type": "text", "text": question})

    check_llama_cpp_requirements()
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    if mmproj_file is not None:
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_file)

    return llava_cpp_messages(model_file, chat_handler, [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": content,
        },
    ], options=options)


def llava_cpp_simple_interrogator_to_json(
        model_file, mmproj_file, system="You are an assistant who perfectly describes images.", question="Describe this image in detail please\nuse json format for output:",
        image=None, schema={}, options={}):

    response_format = {
        "type": "json_object",
        "schema": schema,
    }
    options["response_format"] = response_format
    options["chat_format"] = "chatml"

    result = llava_cpp_simple_interrogator(
        model_file, mmproj_file, system=system, question=question, image=image, options=options)
    return result

