import copy
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

def get_llama_cpp_chat_handlers():
    from llama_cpp import llama_chat_format
    chat_handlers = llama_chat_format.LlamaChatCompletionHandlerRegistry()._chat_handlers
    chat_handlers = list(chat_handlers.keys())

    return chat_handlers


def LlamaCppOptions():
    # chat_handlers = ["auto"] + get_llama_cpp_chat_handlers()
    return {
        # "chat_format": chat_handlers,
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

    model_and_opt = mz_prompt_utils.Utils.cache_get(
        f"llama_cpp_model_and_opt_{model_file}")

    if model_and_opt is None:
        return 0

    model = model_and_opt.get("model")

    del model
    torch.cuda.empty_cache()

    mz_prompt_utils.Utils.cache_set(
        f"llama_cpp_model_and_opt_{model_file}", None)


def llama_cpp_messages(model_file, mmproj_file=None, messages=[], options={}):
    if options is None:
        options = {}
    options = options.copy()
    print(f"Find local model file: {model_file}")
    init_opts = ["n_ctx", "logits_all", "chat_format", "n_gpu_layers"]

    from llama_cpp import Llama
    import llama_cpp

    model_and_opt = mz_prompt_utils.Utils.cache_get(
        f"llama_cpp_model_and_opt_{model_file}")

    is_opts_changed = False

    mz_prompt_utils.Utils.print_log(
        f"llama_cpp_messages chat_format: {options.get('chat_format', None)}")

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

        chat_handler = None
        if mmproj_file is not None:
            # 显存不释放,暂时全局缓存
            chat_handler = mz_prompt_utils.Utils.cache_get(
                f"llama_cpp_messages_mmproj_file_{mmproj_file}"
            )
            if chat_handler is None:
                mz_prompt_utils.Utils.print_log(
                    f"llama_cpp_messages mmproj_file: {mmproj_file}")
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_file)
                mz_prompt_utils.Utils.cache_set(
                    f"llama_cpp_messages_mmproj_file_{mmproj_file}", chat_handler)

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
            "chat_handler": chat_handler,
            "options": options,
        }
        mz_prompt_utils.Utils.cache_set(
            f"llama_cpp_model_and_opt_{model_file}", model_and_opt)

    model = model_and_opt.get("model")
    model.set_seed(options.get("seed", -1))
    model.reset()

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

    try:
        debuf_messages = copy.deepcopy(messages)
        for dindex in range(len(debuf_messages)):
            if debuf_messages[dindex].get("role") == "user":
                debuf_messages_content = debuf_messages[dindex].get(
                    "content", [])
                if type(debuf_messages_content) != list:
                    continue
                for ccindex in range(len(debuf_messages_content)):
                    if debuf_messages_content[ccindex].get("type") == "image_url":
                        debuf_messages[dindex]["content"][ccindex]["image_url"] = debuf_messages[
                            dindex]["content"][ccindex]["image_url"] = None

        mz_prompt_utils.Utils.print_log(
            f"LLAMA_CPP messages: {json.dumps(debuf_messages, indent=4, ensure_ascii=False)}")
    except Exception as e:
        mz_prompt_utils.Utils.print_log(
            f"LLAMA_CPP messages: {messages}")
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
        tools=options.get("tools", None),
        tool_choice=options.get("tool_choice", None),
    )
    mz_prompt_utils.Utils.print_log(f"LLAMA_CPP: \n{output}")
    choices = output.get("choices", [])
    # mz_utils.Utils.print_log(f"LLAMA_CPP choices: \n{choices}")
    if len(choices) == 0:
        return ""

    result = choices[0].get("message", {}).get("content", "")
    return result


def llama_cpp_simple_interrogator_to_json(model_file, use_system=True, system=None, question="", schema={}, options={}):
    options = options.copy()
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

    # if options.get("chat_format", None) is None:
    #     options["chat_format"] = "llama-2"

    result = llama_cpp_messages(model_file, None, messages, options=options)
    result = result.replace("\n", " ")
    return result


def llama_cpp_simple_interrogator(model_file, use_system=True, system=None, question="", options={}):
    if options is None:
        options = {}
    options = options.copy()
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


def llava_cpp_messages(model_file, mmproj_file, messages, options={}):
    if options is None:
        options = {}

    options = options.copy()
    options["logits_all"] = True
    options["n_ctx"] = max(4096, options.get("n_ctx", 4096))

    # if options.get("chat_format", None) is None:
    #     options["chat_format"] = "llama-2"
    return llama_cpp_messages(model_file, mmproj_file, messages, options)


def llava_cpp_simple_interrogator(
        model_file, mmproj_file, system="You are an assistant who perfectly describes images.", question="Describe this image in detail please.",
        image=None, options={}):
    if options is None:
        options = {}
    options = options.copy()

    content = []
    if image is not None:
        data_uri = mz_prompt_utils.Utils.pil_image_to_base64(image)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    content.append({"type": "text", "text": question})

    return llava_cpp_messages(model_file, mmproj_file, [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": content,
        },
    ], options=options)
