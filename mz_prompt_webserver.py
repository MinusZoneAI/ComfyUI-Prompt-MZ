

import asyncio
import uuid


web_msg_pool = {

}


def show_toast_success(message, duration=2000):
    send_message({
        "type": "toast-success",
        "message": message,
        "duration": duration
    })


def send_message(data):
    global web_msg_pool
    for key in web_msg_pool:
        web_msg_pool[key].append(data)


def start_server():
    try:
        global web_msg_pool
        from aiohttp import web
        import server
        app: web.Application = server.PromptServer.instance.app

        async def message(request):
            muuid = uuid.uuid4()
            try:
                ws = web.WebSocketResponse()

                await ws.prepare(request)

                web_msg_pool[muuid] = []
                async for msg in ws:
                    if msg.type == web.WSMsgType.text:
                        if len(web_msg_pool[muuid]) == 0:
                            continue
                        else:
                            await ws.send_json(web_msg_pool[muuid])
                            web_msg_pool[muuid] = []
                    elif msg.type == web.WSMsgType.close:
                        break

                del web_msg_pool[muuid]
                print(f"connection {muuid} closed")
                return ws
            except Exception as e:
                print(e)
                del web_msg_pool[muuid]
                return ws

        if not any([route.get_info().get("path", "") == "/mz_webapi/message" for route in app.router.routes()]):
            print("add route /mz_webapi/message")
            app.router.add_get("/mz_webapi/message", message)
        else:
            print("route /mz_webapi/message is exist")

    except Exception as e:
        print(e)
