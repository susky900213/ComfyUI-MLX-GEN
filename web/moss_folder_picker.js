import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "MlxMossFolderEdit";
const DIRECTORY_WIDGET = "input_directory";
const UPLOAD_ROOT = "mlx_moss";
const IMAGE_EXTENSIONS = new Set(["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]);

function directImageFiles(fileList) {
  return Array.from(fileList || []).filter((file) => {
    const relativePath = String(file.webkitRelativePath || file.name);
    const parts = relativePath.split("/");
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    return parts.length === 2 && IMAGE_EXTENSIONS.has(extension);
  });
}

function safeFolderName(name) {
  const cleaned = String(name || "")
    .replace(/[\\/\u0000-\u001f\u007f]/g, "_")
    .trim();
  return cleaned && cleaned !== "." && cleaned !== ".." ? cleaned : "已选文件夹";
}

function setDirectoryToken(node, widget, token) {
  widget.value = token;
  widget.callback?.(token, app.canvas, node, [0, 0], {});
  node.setDirtyCanvas?.(true, true);
}

function setButtonLabel(node, button, label) {
  button.name = label;
  node.setDirtyCanvas?.(true, true);
}

async function uploadFile(file, subfolder) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("subfolder", subfolder);
  body.append("type", "input");
  body.append("overwrite", "true");

  const response = await api.fetchApi("/upload/image", {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(`上传失败（HTTP ${response.status}）：${file.name}`);
  }
  return response.json();
}

function chooseFolder(node, directoryWidget, button) {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.multiple = true;
  picker.accept = "image/*";
  picker.webkitdirectory = true;
  picker.setAttribute("webkitdirectory", "");
  picker.setAttribute("directory", "");
  picker.addEventListener("change", async () => {
    const files = directImageFiles(picker.files);
    if (!files.length) {
      setButtonLabel(node, button, "选择输入文件夹（未找到图片）");
      return;
    }

    const firstPath = String(files[0].webkitRelativePath || files[0].name);
    const folderName = safeFolderName(firstPath.split("/")[0]);
    const subfolder = `${UPLOAD_ROOT}/${folderName}`;
    setButtonLabel(node, button, `上传中（0/${files.length}）…`);

    try {
      for (let index = 0; index < files.length; index += 1) {
        await uploadFile(files[index], subfolder);
        setButtonLabel(node, button, `上传中（${index + 1}/${files.length}）…`);
      }
      setDirectoryToken(node, directoryWidget, subfolder);
      node.properties = node.properties || {};
      node.properties.mlx_moss_folder_name = folderName;
      node.properties.mlx_moss_image_count = files.length;
      node.properties.mlx_moss_output_directory = `output/moss-folder-edit/${subfolder}`;
      setButtonLabel(node, button, `已选择：${folderName}（${files.length} 张），输出：moss-folder-edit/`);
    } catch (error) {
      setButtonLabel(node, button, "选择输入文件夹（上传失败）");
      console.error("[ComfyUI-MLX-GEN] MOSS 文件夹上传失败", error);
    }
  });
  picker.click();
}

function installFolderPicker(node) {
  if (node?.comfyClass !== TARGET_NODE || node.__mlxMossFolderPicker) return;
  const directoryWidget = node.widgets?.find((widget) => widget.name === DIRECTORY_WIDGET);
  if (!directoryWidget || typeof node.addWidget !== "function") return;

  node.__mlxMossFolderPicker = true;
  // Keep the editable STRING widget visible for manual server-side paths.
  directoryWidget.hidden = false;
  const button = node.addWidget(
    "button",
    "选择输入文件夹",
    "",
    () => chooseFolder(node, directoryWidget, button),
    { serialize: false },
  );

  const directoryIndex = node.widgets.indexOf(directoryWidget);
  const buttonIndex = node.widgets.indexOf(button);
  node.widgets.splice(buttonIndex, 1);
  node.widgets.splice(directoryIndex + 1, 0, button);

  const folderName = node.properties?.mlx_moss_folder_name;
  const imageCount = node.properties?.mlx_moss_image_count;
  if (folderName) {
    setButtonLabel(
      node,
      button,
      `已选择：${folderName}${imageCount ? `（${imageCount} 张）` : ""}，输出：moss-folder-edit/`,
    );
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "ComfyUI-MLX-GEN.MossFolderPicker",
  nodeCreated(node) {
    installFolderPicker(node);
  },
  afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) installFolderPicker(node);
  },
});