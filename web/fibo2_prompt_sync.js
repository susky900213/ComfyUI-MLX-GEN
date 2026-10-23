import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FIBO2_PROMPT_NODES = new Set([
  "MlxFibo2PromptEnhancerT2I",
  "MlxFibo2PromptEnhancerI2I",
]);
const TEXT_ENCODER = "MlxTextEncoder";
const CLEAR_MEMORY_ENDPOINT = "/mlx_gen/fibo2_memory/clear";

function graphLink(graph, linkId) {
  if (!graph?.links || linkId == null) return null;
  return typeof graph.links.get === "function" ? graph.links.get(linkId) : graph.links[linkId];
}

function outputValue(output) {
  const value = output?.vgl_prompt;
  if (Array.isArray(value)) return typeof value[0] === "string" ? value[0] : null;
  return typeof value === "string" ? value : null;
}

function updateConnectedTextWidgets(sourceNode, vglPrompt) {
  const graph = app.graph;
  const links = sourceNode?.outputs?.[0]?.links || [];
  for (const linkId of links) {
    const link = graphLink(graph, linkId);
    const target = graph?.getNodeById?.(Number(link?.target_id ?? link?.targetId));
    if (target?.comfyClass !== TEXT_ENCODER) continue;

    const input = target.inputs?.[Number(link?.target_slot ?? link?.targetSlot)];
    if (input?.name !== "prompt") continue;

    const widget = target.widgets?.find((item) => item?.name === "text");
    if (!widget) continue;
    widget.value = vglPrompt;
    if (widget._state) widget._state.value = vglPrompt;
    widget.callback?.(vglPrompt, app.canvas, target, [0, 0], {});
    target.setDirtyCanvas?.(true, true);
  }
}

function setMemoryButtonLabel(node, button, label) {
  button.name = label;
  node.setDirtyCanvas?.(true, true);
}

async function clearNodeMemory(node, kind, button, defaultLabel) {
  setMemoryButtonLabel(node, button, "清理中…");
  try {
    const response = await api.fetchApi(CLEAR_MEMORY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: String(node.id), kind }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }
    setMemoryButtonLabel(node, button, `${defaultLabel}（已完成）`);
  } catch (error) {
    setMemoryButtonLabel(node, button, `${defaultLabel}（失败）`);
    console.error("[ComfyUI-MLX-GEN] FIBO 2.0 记忆清除失败", error);
  }
}

function installMemoryControls(node) {
  if (!FIBO2_PROMPT_NODES.has(node?.comfyClass) || node.__mlxFibo2MemoryControls) return;
  if (typeof node.addWidget !== "function" || !Array.isArray(node.widgets)) return;

  node.__mlxFibo2MemoryControls = true;
  const contextButton = node.addWidget(
    "button",
    "清除上下文",
    "",
    () => clearNodeMemory(node, "context", contextButton, "清除上下文"),
    { serialize: false },
  );
  const memoryButton = node.addWidget(
    "button",
    "删除长期记忆",
    "",
    () => clearNodeMemory(node, "memory", memoryButton, "删除长期记忆"),
    { serialize: false },
  );
  node.setDirtyCanvas?.(true, true);
}

api.addEventListener("executed", ({ detail }) => {
  const sourceNode = app.graph?.getNodeById?.(Number(detail?.node));
  if (!sourceNode || !FIBO2_PROMPT_NODES.has(sourceNode.comfyClass)) return;
  const vglPrompt = outputValue(detail?.output);
  if (vglPrompt == null) return;
  updateConnectedTextWidgets(sourceNode, vglPrompt);
});

app.registerExtension({
  name: "ComfyUI-MLX-GEN.Fibo2PromptSync",
  nodeCreated(node) {
    installMemoryControls(node);
  },
  afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) installMemoryControls(node);
  },
});