import { McpServer } from "@modelcontextprotocol/server";
import * as z from "zod/v4";
import { animations } from "../src/animations";
import { describeScene, findScene, validateParameters } from "../src/control";
import { version } from "../package.json";
import { BrowserBridge } from "./bridge";

export { BrowserBridge } from "./bridge";

export function createMcpServer(bridge: BrowserBridge) {
  const server = new McpServer({ name: "vj-web", version });
  const result = (value: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }] });
  const safe = (action: () => unknown | Promise<unknown>) => Promise.resolve().then(action).then(result).catch((error) => ({
    isError: true, content: [{ type: "text" as const, text: error instanceof Error ? error.message : String(error) }],
  }));
  const readOnly = { readOnlyHint: true, destructiveHint: false, openWorldHint: false };
  const edit = { readOnlyHint: false, destructiveHint: false, openWorldHint: false };
  server.registerTool("get_connection", {
    description: "Get the local browser connection URL and video export folder. Open the URL in Chrome or Edge to enable live control and MP4 export.",
    inputSchema: z.object({}), annotations: readOnly,
  }, () => safe(() => bridge.connection));
  server.registerTool("list_scenes", {
    description: "List every available vj-web scene, optionally filtering by name or description. Works without a browser.",
    inputSchema: z.object({ query: z.string().optional() }), annotations: readOnly,
  }, ({ query }) => safe(() => animations.filter((scene) =>
    `${scene.id} ${scene.name} ${scene.description}`.toLowerCase().includes((query ?? "").toLowerCase()),
  ).map(({ id, name, description, stateful }) => ({ id, name, description, requiresFloatTargets: stateful ?? false }))));
  server.registerTool("get_scene", {
    description: "Get a scene's parameter IDs, types, defaults and valid ranges. Seeds are randomized per browser session.",
    inputSchema: z.object({ sceneId: z.string() }), annotations: readOnly,
  }, ({ sceneId }) => safe(() => describeScene(findScene(animations, sceneId))));
  server.registerTool("get_state", {
    description: "Read the connected browser's active scene, parameter values and rendering capabilities.",
    inputSchema: z.object({}), annotations: readOnly,
  }, () => safe(() => bridge.request("get_state")));
  server.registerTool("select_scene", {
    description: "Switch the connected browser to a scene. Scene changes are blocked during video export.",
    inputSchema: z.object({ sceneId: z.string() }), annotations: { ...edit, idempotentHint: false },
  }, ({ sceneId }) => safe(() => {
    findScene(animations, sceneId);
    return bridge.request("select_scene", { sceneId });
  }));
  server.registerTool("set_parameters", {
    description: "Set parameters on the active scene. Supply its sceneId to prevent editing a scene changed by the user. Values are validated as a batch; seed changes restart simulations.",
    inputSchema: z.object({ sceneId: z.string(), values: z.record(z.string(), z.number()) }), annotations: edit,
  }, ({ sceneId, values }) => safe(() => {
    validateParameters(findScene(animations, sceneId), values);
    return bridge.request("set_parameters", { sceneId, values });
  }));
  for (const command of ["reset_scene", "reseed_scene"] as const) {
    server.registerTool(command, {
      description: command === "reset_scene" ? "Reset the active scene to its browser-session defaults." : "Randomize the active scene's seeds and restart its simulation.",
      inputSchema: z.object({ sceneId: z.string() }), annotations: edit,
    }, ({ sceneId }) => safe(() => {
      findScene(animations, sceneId);
      return bridge.request(command, { sceneId });
    }));
  }
  server.registerTool("export_video", {
    description: "Start an asynchronous silent H.264 MP4 export of the active scene at its current settings, starting at time zero. No save dialog. Returns a job ID; poll get_export_status until completed before using the file. Keep the browser tab open. Only one export at a time.",
    inputSchema: z.object({
      duration: z.number().min(1).max(3600).default(10).describe("Duration in seconds"),
      fps: z.union([z.literal(30), z.literal(60)]).default(60),
      width: z.number().int().min(16).max(3840).multipleOf(2).default(1920),
      height: z.number().int().min(16).max(2160).multipleOf(2).default(1080),
    }), annotations: edit,
  }, (options) => safe(() => bridge.startExport(options)));
  server.registerTool("get_export_status", {
    description: "Get an export's status, frame progress, bytes written, output path and any error. Only completed files are ready to use.",
    inputSchema: z.object({ jobId: z.string() }), annotations: readOnly,
  }, ({ jobId }) => safe(() => bridge.exportStatus(jobId)));
  server.registerTool("cancel_export", {
    description: "Cancel an unfinished export and delete its incomplete file. Completed videos are kept.",
    inputSchema: z.object({ jobId: z.string() }), annotations: edit,
  }, ({ jobId }) => safe(() => bridge.cancelExport(jobId)));
  server.registerResource("scenes", "vj-web://scenes", {
    description: "The vj-web scene catalog and control ranges", mimeType: "application/json",
  }, (uri) => ({ contents: [{ uri: uri.href, mimeType: "application/json", text: JSON.stringify(animations.map(describeScene)) }] }));
  return server;
}
