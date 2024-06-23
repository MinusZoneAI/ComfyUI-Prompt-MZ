import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function showToast(message, duration = 3000) {
  const toast = document.createElement("div");
  toast.style.position = "fixed";
  toast.style.top = "20px";
  toast.style.left = "50%";
  toast.style.transform = "translateX(-50%)";
  toast.style.padding = "10px 20px";
  toast.style.backgroundColor = "var(--comfy-menu-bg)";
  toast.style.color = "var(--input-text)";
  toast.style.borderRadius = "10px";
  toast.style.border = "2px solid var(--border-color)";
  toast.style.zIndex = "9999";

  toast.textContent = message;
  document.body.appendChild(toast);
  await sleep(duration);
  toast.remove();
}

async function waitMessage() {
  var websocket = new WebSocket(
    `ws://${window.location.host}/mz_webapi/message`
  );
  websocket.onmessage = async (event) => {
    const resp = JSON.parse(event.data);
    console.log("Message received", resp);

    for (const data of resp) {
      if (data.type === "toast-success") {
        await showToast(data.message, data?.duration || 3000);
      }
    }
  };
  websocket.onclose = async (event) => {
    console.log("Connection closed", event);
  };

  websocket.onerror = async (event) => {
    console.log("Connection error", event);
  };

  // for (;;) {
  //   await sleep(1000);
  //   try {
  //     if (websocket.readyState !== WebSocket.OPEN) {
  //       return;
  //     }
  //     websocket.send(
  //       JSON.stringify({
  //         type: "ping",
  //       })
  //     );
  //   } catch (error) {
  //     return;
  //   }
  // }
}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const my_ui = {
  name: "prompt_mz.ui",
  setup() {},
  init: async () => {
    console.log("prompt_mz Registering UI extension");

    waitMessage();
  },

  /**
   * @param {import("./types/comfy").NodeType} nodeType
   * @param {import("./types/comfy").NodeDef} nodeData
   * @param {import("./types/comfy").App} app
   */
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    switch (nodeData.name) {
      case "MZ_OpenAIApiCLIPTextEncode":
      case "MZ_LLama3CLIPTextEncode":
      case "MZ_Phi3CLIPTextEncode":
      case "MZ_BaseLLamaCPPCLIPTextEncode":
      case "MZ_LLavaImageInterrogator":
      case "MZ_BaseLLavaImageInterrogator":
      case "MZ_LLamaCPPCLIPTextEncode":
      case "MZ_ImageInterrogatorCLIPTextEncode":
      case "MZ_Florence2CLIPTextEncode":
      case "MZ_PaliGemmaCLIPTextEncode":
        // Node Created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const ret = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined;

          console.log("onNodeCreated:", this);
          const nodeName = this.name + "_" + "customtext";
          const wi = ComfyWidgets.STRING(
            this,
            nodeName,
            [
              "STRING",
              {
                default: "",
                placeholder: "Text message output...",
                multiline: true,
              },
            ],
            app
          );
          wi.widget.inputEl.readOnly = true;

          return ret;
        };

        const outSet = function (texts) {
          if (texts.length > 0) {
            let widget_id = this?.widgets.findIndex(
              (w) => w.name === this.name + "_" + "customtext"
            );
            if (Array.isArray(texts))
              texts = texts
                .filter((word) => word.trim() !== "")
                .map((word) => word.trim())
                .join(" ");
            this.widgets[widget_id].value = texts;
            app.graph.setDirtyCanvas(true);
          }
        };

        // onConfigure
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (w) {
          onConfigure?.apply(this, arguments);

          // outSet.call(this, a?.string);
        };

        // onExecuted
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (a, b) {
          // console.log("onExecuted:", arguments);
          onExecuted?.apply(this, arguments);

          outSet.call(this, a?.string);
        };
    }
  },
};

app.registerExtension(my_ui);
