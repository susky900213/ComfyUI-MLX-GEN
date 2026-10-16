import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "MlxDownloadScrambledHtmlImages";
const IMAGE_DIRECTORY_WIDGET = "image_directory";
const HTML_FILE_WIDGET = "html_file";
const INPUT_PREFIX = "__comfy_input__/";
const UPLOAD_ROOT = "mlx_html_scramble";
const IMAGE_EXTENSIONS = new Set(["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]);

function safePathPart(value, fallback) {
  const cleaned = String(value || "")
    .replace(/[\\/\u0000-\u001f\u007f]/g, "_")
    .trim();
  return cleaned && cleaned !== "." && cleaned !== ".." ? cleaned : fallback;
}

function directImageFiles(fileList) {
  return Array.from(fileList || []).filter((file) => {
    const relativePath = String(file.webkitRelativePath || file.name);
    const parts = relativePath.split("/");
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    return parts.length === 2 && IMAGE_EXTENSIONS.has(extension);
  });
}

function setWidgetValue(node, widget, value) {
  widget.value = value;
  widget.callback?.(value, app.canvas, node, [0, 0], {});
  node.setDirtyCanvas?.(true, true);
}

function setButtonLabel(node, button, label) {
  button.name = label;
  node.setDirtyCanvas?.(true, true);
}

async function uploadFile(file, subfolder) {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("subfolder", subfolder);
  const response = await api.fetchApi("/mlx_gen/html_scramble_upload", {
    method: "POST",
    body,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`上传失败（HTTP ${response.status}）：${message || file.name}`);
  }
  return response.json();
}

function createFilePicker({ accept, directory = false }) {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.multiple = directory;
  if (accept) picker.accept = accept;
  if (directory) {
    picker.webkitdirectory = true;
    picker.setAttribute("webkitdirectory", "");
    picker.setAttribute("directory", "");
  }
  return picker;
}

function chooseImageDirectory(node, widget, button) {
  const picker = createFilePicker({ accept: "image/*", directory: true });
  picker.addEventListener("change", async () => {
    const files = directImageFiles(picker.files);
    if (!files.length) {
      setButtonLabel(node, button, "选择图片文件夹（未找到支持的图片）");
      return;
    }

    const firstPath = String(files[0].webkitRelativePath || files[0].name);
    const folderName = safePathPart(firstPath.split("/")[0], "images");
    const subfolder = `${UPLOAD_ROOT}/${folderName}`;
    setButtonLabel(node, button, `上传图片中（0/${files.length}）…`);

    try {
      for (let index = 0; index < files.length; index += 1) {
        await uploadFile(files[index], subfolder);
        setButtonLabel(node, button, `上传图片中（${index + 1}/${files.length}）…`);
      }
      setWidgetValue(node, widget, `${INPUT_PREFIX}${subfolder}`);
      node.properties = node.properties || {};
      node.properties.mlx_html_scramble_image_folder_name = folderName;
      node.properties.mlx_html_scramble_image_count = files.length;
      setButtonLabel(node, button, `已选择图片：${folderName}（${files.length} 张）`);
    } catch (error) {
      setButtonLabel(node, button, "选择图片文件夹（上传失败）");
      console.error("[ComfyUI-MLX-GEN] HTML 图片文件夹上传失败", error);
    }
  });
  picker.click();
}

function chooseHtmlFile(node, widget, button) {
  const picker = createFilePicker({ accept: ".html,.htm,text/html" });
  picker.addEventListener("change", async () => {
    const file = picker.files?.[0];
    if (!file) return;

    const filename = safePathPart(file.name, "page.html");
    const subfolder = `${UPLOAD_ROOT}/html`;
    setButtonLabel(node, button, "上传 HTML 页面中…");
    try {
      await uploadFile(file, subfolder);
      setWidgetValue(node, widget, `${INPUT_PREFIX}${subfolder}/${filename}`);
      node.properties = node.properties || {};
      node.properties.mlx_html_scramble_html_file_name = filename;
      setButtonLabel(node, button, `已选择 HTML：${filename}`);
    } catch (error) {
      setButtonLabel(node, button, "选择 HTML 页面（上传失败）");
      console.error("[ComfyUI-MLX-GEN] HTML 页面上传失败", error);
    }
  });
  picker.click();
}

function restoreLabel(node, button, widgetName) {
  const widget = node.widgets?.find((item) => item.name === widgetName);
  const value = String(widget?.value || "");
  if (!value || value.startsWith("<")) return;

  if (widgetName === IMAGE_DIRECTORY_WIDGET) {
    const folder = node.properties?.mlx_html_scramble_image_folder_name;
    const count = node.properties?.mlx_html_scramble_image_count;
    setButtonLabel(node, button, `已选择图片：${folder || value.split("/").pop()}${count ? `（${count} 张）` : ""}`);
  } else if (widgetName === HTML_FILE_WIDGET) {
    const filename = node.properties?.mlx_html_scramble_html_file_name || value.split("/").pop();
    setButtonLabel(node, button, `已选择 HTML：${filename}`);
  }
}

function installPicker(node) {
  if (node?.comfyClass !== TARGET_NODE || node.__mlxHtmlScramblePicker) return;
  if (typeof node.addWidget !== "function" || !Array.isArray(node.widgets)) return;

  const definitions = [
    [IMAGE_DIRECTORY_WIDGET, "选择图片文件夹", (widget, button) => chooseImageDirectory(node, widget, button)],
    [HTML_FILE_WIDGET, "选择 HTML 页面", (widget, button) => chooseHtmlFile(node, widget, button)],
  ];
  const controls = [];
  for (const [widgetName, label, onClick] of definitions) {
    const widget = node.widgets.find((item) => item.name === widgetName);
    if (!widget) return;
    widget.hidden = true;
    const button = node.addWidget("button", label, "", () => onClick(widget, button), { serialize: false });
    controls.push({ widget, button });
  }

  node.__mlxHtmlScramblePicker = true;
  for (const { widget, button } of controls) {
    const widgetIndex = node.widgets.indexOf(widget);
    const buttonIndex = node.widgets.indexOf(button);
    node.widgets.splice(buttonIndex, 1);
    node.widgets.splice(widgetIndex, 0, button);
    restoreLabel(node, button, widget.name);
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "ComfyUI-MLX-GEN.HtmlScrambleImagePicker",
  nodeCreated(node) {
    installPicker(node);
  },
  afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) installPicker(node);
  },
});