import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PE_NODES = new Set(["MlxQwenImagePET2I", "MlxQwenImagePEI2I"]);
const TEXT_ENCODER = "MlxTextEncoder";

function graphLink(graph, linkId) {
  if (!graph?.links || linkId == null) return null;
  return typeof graph.links.get === "function" ? graph.links.get(linkId) : graph.links[linkId];
}

function outputValue(output) {
  const value = output?.rewritten_prompt;
  if (Array.isArray(value)) return typeof value[0] === "string" ? value[0] : null;
  return typeof value === "string" ? value : null;
}

function updateConnectedTextWidgets(sourceNode, rewrittenPrompt) {
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
    widget.value = rewrittenPrompt;
    if (widget._state) widget._state.value = rewrittenPrompt;
    widget.callback?.(rewrittenPrompt, app.canvas, target, [0, 0], {});
    target.setDirtyCanvas?.(true, true);
  }
}

api.addEventListener("executed", ({ detail }) => {
  const sourceNode = app.graph?.getNodeById?.(Number(detail?.node));
  if (!sourceNode || !PE_NODES.has(sourceNode.comfyClass)) return;

  const rewrittenPrompt = outputValue(detail?.output);
  if (rewrittenPrompt == null) return;
  updateConnectedTextWidgets(sourceNode, rewrittenPrompt);
});

app.registerExtension({
  name: "ComfyUI-MLX-GEN.QwenImagePEPromptSync",
});
