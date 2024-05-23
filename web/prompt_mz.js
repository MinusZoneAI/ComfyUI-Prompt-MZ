import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const my_ui = {
  name: "prompt_mz.ui",
  setup() {},
  init: async () => {
    console.log("prompt_mz Registering UI extension");
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
        // Node Created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const ret = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined;

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
        };

        // onExecuted
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (a, b) {
          console.log("onExecuted:", arguments);
          onExecuted?.apply(this, arguments);

          outSet.call(this, a?.string);
        };
    }
  },
};

app.registerExtension(my_ui);
