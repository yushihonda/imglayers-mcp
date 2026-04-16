"""Static HTML viewer for a decomposed project.

Generates `preview/index.html` — a single self-contained file that uses the
relative asset paths already produced by the pipeline. Opens cleanly via
`file://` in any browser; no server required.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ..storage.paths import ProjectPaths


def render_viewer_html(manifest: dict, paths: ProjectPaths) -> Path:
    out = paths.preview_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    project_id = manifest.get("projectId", "")
    canvas = manifest.get("canvas", {}) or {}
    canvas_w = int(canvas.get("width") or 0)
    canvas_h = int(canvas.get("height") or 0)
    layers = manifest.get("layers", []) or []

    # Asset paths relative to preview/index.html.
    original_src = "../meta/original.png"
    composed_src = "preview.png"
    annotated_src = "preview_annotated.png"
    grid_src = "preview_grid.png"

    groups = manifest.get("groups", []) or []

    layer_records = []
    for L in layers:
        bbox = L.get("bbox") or {}
        asset_rel = (L.get("assets") or {}).get("png") or f"layers/{L['id']}.png"
        text = L.get("text") or {}
        style = L.get("styleHints") or {}
        layer_records.append(
            {
                "id": L.get("id"),
                "name": L.get("name") or L.get("id"),
                "type": L.get("type"),
                "role": L.get("semanticRole"),
                "zIndex": L.get("zIndex", 0),
                "bbox": {
                    "x": float(bbox.get("x", 0)),
                    "y": float(bbox.get("y", 0)),
                    "w": float(bbox.get("width", 0)),
                    "h": float(bbox.get("height", 0)),
                },
                "src": f"../{asset_rel}",
                "textContent": text.get("content"),
                "textColor": style.get("textColor"),
                "fontSize": style.get("fontSize"),
                "fontWeight": style.get("fontWeight"),
                "fill": style.get("fill"),
                "parentGroup": (L.get("children") or [None])[0],
            }
        )

    group_records = []
    for G in groups:
        child_ids = G.get("layerIds") or G.get("layer_ids") or []
        child_bboxes = []
        for lr in layer_records:
            if lr["id"] in child_ids:
                child_bboxes.append(lr["bbox"])
        gbbox = None
        if child_bboxes:
            gbbox = {
                "x": min(b["x"] for b in child_bboxes),
                "y": min(b["y"] for b in child_bboxes),
                "w": max(b["x"] + b["w"] for b in child_bboxes) - min(b["x"] for b in child_bboxes),
                "h": max(b["y"] + b["h"] for b in child_bboxes) - min(b["y"] for b in child_bboxes),
            }
        group_records.append(
            {
                "id": G.get("id"),
                "name": G.get("name", ""),
                "childIds": child_ids,
                "bbox": gbbox,
            }
        )

    data_json = json.dumps(
        {
            "projectId": project_id,
            "canvas": {"width": canvas_w, "height": canvas_h},
            "background": canvas.get("background"),
            "layers": layer_records,
            "groups": group_records,
            "images": {
                "original": original_src,
                "composed": composed_src,
                "annotated": annotated_src,
                "grid": grid_src,
            },
        },
        ensure_ascii=False,
    )

    out.write_text(_TEMPLATE.replace("__DATA__", data_json).replace("__TITLE__", html.escape(project_id)), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>imglayers · __TITLE__</title>
<style>
  :root { color-scheme: dark; --bg0: #1e1e1e; --bg1: #252526; --bg2: #2d2d2d;
          --fg: #cccccc; --fg2: #999; --accent: #0d99ff; --accent-bg: rgba(13,153,255,.12);
          --border: #3c3c3c; --hover: #2a2d2e; --selected: #04395e; }
  * { box-sizing: border-box; margin: 0; }
  body { font: 12px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background: var(--bg0); color: var(--fg); display: grid;
         grid-template-columns: 260px 1fr 280px; grid-template-rows: 42px 1fr;
         height: 100vh; overflow: hidden; }

  /* ---- Header bar ---- */
  header { grid-column: 1 / -1; display: flex; align-items: center; gap: 10px;
           padding: 0 12px; border-bottom: 1px solid var(--border); background: var(--bg1); }
  header h1 { font-size: 13px; font-weight: 600; white-space: nowrap; }
  header .pid { font-size: 11px; opacity: .55; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
  .tabs { display: flex; gap: 2px; margin-left: auto; }
  .tab { padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px;
         color: var(--fg2); transition: .1s; }
  .tab:hover { background: var(--bg2); color: var(--fg); }
  .tab.active { background: var(--accent); color: #fff; }

  /* ---- Left panel: layer tree ---- */
  .panel-left { border-right: 1px solid var(--border); background: var(--bg1);
                display: flex; flex-direction: column; overflow: hidden; }
  .panel-left .section-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
                               letter-spacing: .04em; padding: 10px 12px 6px; color: var(--fg2); }
  .panel-left .toolbar { display: flex; gap: 4px; padding: 0 10px 8px; }
  .panel-left .toolbar button { flex: 1; padding: 3px; background: var(--bg2); color: var(--fg2);
                                border: 1px solid var(--border); border-radius: 4px; cursor: pointer;
                                font-size: 10px; transition: .1s; }
  .panel-left .toolbar button:hover { background: var(--hover); color: var(--fg); }
  #layer-tree { flex: 1; overflow-y: auto; padding-bottom: 20px; }

  /* Tree node rows */
  .tree-node { display: flex; align-items: center; height: 28px; padding-right: 8px;
               cursor: pointer; user-select: none; transition: background .08s; }
  .tree-node:hover { background: var(--hover); }
  .tree-node.selected { background: var(--selected); }
  .tree-node .indent { flex-shrink: 0; position: relative; }
  .tree-node .indent::before { content: ""; position: absolute; top: 0; bottom: 0;
                               right: 4px; border-left: 1px dotted var(--border); opacity: .6; }
  .tree-node .arrow { width: 16px; height: 16px; display: flex; align-items: center;
                      justify-content: center; font-size: 9px; color: var(--fg2);
                      flex-shrink: 0; transition: transform .12s; }
  .tree-node .arrow.open { transform: rotate(90deg); }
  .tree-node .arrow.hidden { visibility: hidden; }
  .tree-node .icon { width: 16px; height: 16px; margin-right: 6px; flex-shrink: 0;
                     display: flex; align-items: center; justify-content: center; font-size: 11px; }
  .tree-node .node-name { flex: 1; font-size: 12px; white-space: nowrap; overflow: hidden;
                          text-overflow: ellipsis;
                          font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
  .tree-node .node-type { font-size: 10px; color: var(--fg2); margin-left: 4px; flex-shrink: 0; }
  .tree-node .vis-toggle { width: 18px; height: 18px; display: flex; align-items: center;
                           justify-content: center; opacity: 0; font-size: 12px; flex-shrink: 0;
                           border-radius: 3px; transition: .1s; }
  .tree-node:hover .vis-toggle, .tree-node .vis-toggle.off { opacity: 1; }
  .tree-node .vis-toggle:hover { background: var(--bg2); }
  .tree-node .vis-toggle.off { color: var(--fg2); }
  .group-children { overflow: hidden; }
  .group-children.collapsed { display: none; }

  /* ---- Center: canvas stage ---- */
  .stage { overflow: auto; display: flex; align-items: flex-start; justify-content: center;
           padding: 24px; background: var(--bg0); }
  .stage-inner { position: relative; box-shadow: 0 2px 24px rgba(0,0,0,.5); border-radius: 2px; }
  .layer-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
               pointer-events: none; object-fit: fill; }
  .bbox-hl { position: absolute; border: 1.5px solid var(--accent); background: var(--accent-bg);
             pointer-events: none; border-radius: 1px; }
  .bbox-label { position: absolute; top: -18px; left: 0; font-size: 10px; padding: 1px 4px;
                background: var(--accent); color: #fff; border-radius: 2px; white-space: nowrap; }
  .stage-note { padding: 24px; opacity: .4; }

  /* ---- Right panel: inspector ---- */
  .panel-right { border-left: 1px solid var(--border); background: var(--bg1);
                 overflow-y: auto; padding: 12px; }
  .panel-right .section-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
                                letter-spacing: .04em; color: var(--fg2); margin-bottom: 8px; }
  .inspector-empty { color: var(--fg2); font-size: 12px; padding: 8px 0; }
  .prop-grid { display: grid; grid-template-columns: 80px 1fr; gap: 4px 8px; font-size: 12px; }
  .prop-grid .label { color: var(--fg2); }
  .prop-grid .value { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; word-break: break-all; }
  .color-chip { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
                border: 1px solid var(--border); vertical-align: middle; margin-right: 4px; }
  .inspector-thumb { max-width: 100%; border-radius: 4px; margin-top: 8px; background: var(--bg2); }
</style>
</head>
<body>
<header>
  <h1>imglayers</h1>
  <span class="pid" id="pid"></span>
  <div class="tabs" id="tabs"></div>
</header>

<div class="panel-left">
  <div class="section-title">Layers</div>
  <div class="toolbar">
    <button id="all-on">Show all</button>
    <button id="all-off">Hide all</button>
    <button id="expand-all">Expand</button>
    <button id="collapse-all">Collapse</button>
  </div>
  <div id="layer-tree"></div>
</div>

<div class="stage" id="stage"></div>

<div class="panel-right">
  <div class="section-title">Inspector</div>
  <div id="inspector"><div class="inspector-empty">Select a layer</div></div>
</div>

<script>
const DATA = __DATA__;
const S = {
  view: "interactive",
  visible: new Set(DATA.layers.map(l => l.id)),
  selected: null,
  expanded: new Set((DATA.groups || []).map(g => g.id)),
};
const layerById = Object.fromEntries(DATA.layers.map(l => [l.id, l]));
const groupById = Object.fromEntries((DATA.groups || []).map(g => [g.id, g]));
const childToGroup = {};
(DATA.groups || []).forEach(g => g.childIds.forEach(cid => { childToGroup[cid] = g.id; }));

document.getElementById("pid").textContent = DATA.projectId;

/* ---- Icons ---- */
const ICONS = {
  background: "&#9645;", image: "&#9724;", text: "T", vector_like: "&#9674;",
  unknown: "?", group: "&#9660;",
};
function typeIcon(type) { return ICONS[type] || ICONS.unknown; }

/* ---- Tabs ---- */
const TABS = [
  { id: "interactive", label: "Interactive" }, { id: "original", label: "Original" },
  { id: "composed", label: "Composed" }, { id: "annotated", label: "Annotated" },
  { id: "grid", label: "Grid" },
];
const tabsEl = document.getElementById("tabs");
TABS.forEach(t => {
  const b = document.createElement("div");
  b.className = "tab" + (t.id === S.view ? " active" : "");
  b.textContent = t.label;
  b.dataset.view = t.id;
  b.onclick = () => { S.view = t.id; renderTabs(); renderStage(); };
  tabsEl.appendChild(b);
});
function renderTabs() {
  tabsEl.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("active", b.dataset.view === S.view));
}

/* ---- Layer tree ---- */
const treeEl = document.getElementById("layer-tree");

function buildTree() {
  const groups = DATA.groups || [];
  const containerGroups = groups.filter(g => g.id.startsWith("container_"));
  const otherGroups = groups.filter(g => !g.id.startsWith("container_"));

  const containerParents = new Set(containerGroups.map(g => g.childIds[0]));
  const containerGroupByParentId = {};
  containerGroups.forEach(g => { containerGroupByParentId[g.childIds[0]] = g; });

  // A layer is a "non-parent child of a container" if it appears in a
  // container group's layerIds but not as the first (parent) entry.
  const absorbedByContainer = new Set();
  containerGroups.forEach(g => {
    g.childIds.slice(1).forEach(id => absorbedByContainer.add(id));
  });

  // Non-container groups (textgroup / sibling_cluster / row_cluster /
  // container_group's synthetic sub-groups) are attached to their members.
  const groupsContainingLayer = {};
  otherGroups.forEach(g => {
    g.childIds.forEach(id => {
      if (!groupsContainingLayer[id]) groupsContainingLayer[id] = [];
      groupsContainingLayer[id].push(g);
    });
  });

  function wrapLayer(layer, visited) {
    // If this layer is a container parent, return a container node.
    if (containerParents.has(layer.id) && !visited.has(layer.id)) {
      visited.add(layer.id);
      const g = containerGroupByParentId[layer.id];
      const subItems = g.childIds.slice(1)
        .map(id => resolveChildItem(id, visited))
        .filter(Boolean);
      return { kind: "container", data: layer, group: g, children: subItems };
    }
    return { kind: "layer", data: layer };
  }

  const groupIndex = Object.fromEntries(groups.map(g => [g.id, g]));

  function resolveChildItem(childId, visited) {
    if (layerById[childId]) return wrapLayer(layerById[childId], visited);
    const g = groupIndex[childId];
    if (!g) return null;
    return wrapGroup(g, visited);
  }

  function wrapGroup(group, visited) {
    if (visited.has(group.id)) return null;
    visited.add(group.id);
    const children = group.childIds.map(id => {
      // Inside a non-container group, each child id is a layer id.
      if (layerById[id]) return wrapLayer(layerById[id], visited);
      return resolveChildItem(id, visited);
    }).filter(Boolean);
    return { kind: "group", data: group, children };
  }

  // Root level: layers not absorbed by a container AND not already inside
  // a non-container group whose other members pull them in.
  const tree = [];
  const visited = new Set();

  // First pass: synthetic groups whose children are all "top-level" go to root.
  const rootGroupIds = new Set();
  otherGroups.forEach(g => {
    const kids = g.childIds.filter(id => layerById[id]);
    if (kids.length === 0) return;
    const allTop = kids.every(id => !absorbedByContainer.has(id));
    if (allTop && (g.id.startsWith("row_") || g.id.startsWith("cluster_"))) {
      rootGroupIds.add(g.id);
    }
  });

  const inRootGroup = new Set();
  rootGroupIds.forEach(gid => {
    const g = groups.find(gg => gg.id === gid);
    if (!g) return;
    g.childIds.forEach(id => inRootGroup.add(id));
  });

  rootGroupIds.forEach(gid => {
    const g = groups.find(gg => gg.id === gid);
    if (g) {
      const item = wrapGroup(g, visited);
      if (item) tree.push(item);
    }
  });

  // Remaining top-level layers.
  const rootLayers = DATA.layers
    .filter(L => !absorbedByContainer.has(L.id) && !inRootGroup.has(L.id))
    .sort((a, b) => b.zIndex - a.zIndex);
  rootLayers.forEach(L => {
    if (visited.has(L.id)) return;
    tree.push(wrapLayer(L, visited));
  });

  return tree;
}

function renderTree() {
  treeEl.innerHTML = "";
  const tree = buildTree();
  tree.forEach(node => renderNode(treeEl, node, 0));
}

function renderNode(parent, node, depth) {
  if (!node) return;
  if (node.kind === "layer") {
    renderLayerNode(parent, node.data, depth);
  } else if (node.kind === "group") {
    renderGroupNode(parent, node.data, node.children, depth);
  } else if (node.kind === "container") {
    renderContainerNode(parent, node.data, node.group, node.children, depth);
  }
}

function renderContainerNode(parent, parentLayer, group, childItems, depth) {
  // Render the container parent as a "frame" row.
  const row = document.createElement("div");
  row.className = "tree-node" + (S.selected === parentLayer.id ? " selected" : "");
  const iconColor = parentLayer.role === "card" ? "#98c379" :
                    parentLayer.role === "button" ? "#e5c07b" : "#abb2bf";
  row.innerHTML =
    `<span class="indent" style="width:${depth*16+8}px"></span>` +
    `<span class="arrow ${S.expanded.has(group.id)?'open':''}">&#9654;</span>` +
    `<span class="icon" style="color:${iconColor}">&#9633;</span>` +
    `<span class="node-name">${esc(parentLayer.name || parentLayer.id)}</span>` +
    `<span class="node-type">${parentLayer.role || parentLayer.type}</span>` +
    `<span class="vis-toggle ${S.visible.has(parentLayer.id)?'':'off'}">${S.visible.has(parentLayer.id)?'&#128065;':'&#128064;'}</span>`;

  row.querySelector(".arrow").onclick = e => {
    e.stopPropagation();
    S.expanded.has(group.id) ? S.expanded.delete(group.id) : S.expanded.add(group.id);
    renderTree();
  };
  row.querySelector(".vis-toggle").onclick = e => {
    e.stopPropagation();
    if (S.visible.has(parentLayer.id)) S.visible.delete(parentLayer.id);
    else S.visible.add(parentLayer.id);
    renderTree(); renderStage();
  };
  row.onclick = () => {
    S.selected = S.selected === parentLayer.id ? null : parentLayer.id;
    renderTree(); renderStage(); renderInspector();
  };
  parent.appendChild(row);

  const childWrap = document.createElement("div");
  childWrap.className = "group-children" + (S.expanded.has(group.id) ? "" : " collapsed");
  childItems.forEach(item => renderNode(childWrap, item, depth + 1));
  parent.appendChild(childWrap);
}

function groupKindColor(id) {
  if (id.startsWith("textgroup_")) return "#c49cde";
  if (id.startsWith("row_")) return "#61afef";
  if (id.startsWith("cluster_")) return "#98c379";
  return "#c49cde";
}

function collectGroupLeafIds(node, out) {
  if (!node) return out;
  if (node.kind === "layer") { out.add(node.data.id); return out; }
  if (node.kind === "container") {
    out.add(node.data.id);
    node.children.forEach(c => collectGroupLeafIds(c, out));
    return out;
  }
  if (node.kind === "group") {
    node.children.forEach(c => collectGroupLeafIds(c, out));
  }
  return out;
}

function renderGroupNode(parent, group, children, depth) {
  const row = document.createElement("div");
  row.className = "tree-node" + (S.selected === group.id ? " selected" : "");
  const leafIds = Array.from(collectGroupLeafIds({kind:"group", data:group, children}, new Set()));
  const allOn = leafIds.length > 0 && leafIds.every(id => S.visible.has(id));
  row.innerHTML =
    `<span class="indent" style="width:${depth*16+8}px"></span>` +
    `<span class="arrow ${S.expanded.has(group.id)?'open':''}">&#9654;</span>` +
    `<span class="icon" style="color:${groupKindColor(group.id)}">${ICONS.group}</span>` +
    `<span class="node-name">${esc(group.name || group.id)}</span>` +
    `<span class="node-type">${children.length}</span>` +
    `<span class="vis-toggle ${allOn?'':'off'}">${allOn?'&#128065;':'&#128064;'}</span>`;

  row.querySelector(".arrow").onclick = e => {
    e.stopPropagation();
    S.expanded.has(group.id) ? S.expanded.delete(group.id) : S.expanded.add(group.id);
    renderTree();
  };
  row.querySelector(".vis-toggle").onclick = e => {
    e.stopPropagation();
    leafIds.forEach(id => allOn ? S.visible.delete(id) : S.visible.add(id));
    renderTree(); renderStage();
  };
  row.onclick = () => { S.selected = S.selected === group.id ? null : group.id; renderTree(); renderStage(); renderInspector(); };
  parent.appendChild(row);

  const childWrap = document.createElement("div");
  childWrap.className = "group-children" + (S.expanded.has(group.id) ? "" : " collapsed");
  children.forEach(item => renderNode(childWrap, item, depth + 1));
  parent.appendChild(childWrap);
}

function renderLayerNode(parent, L, depth) {
  const row = document.createElement("div");
  row.className = "tree-node" + (S.selected === L.id ? " selected" : "");
  const iconColor = L.type === "text" ? "#56b6c2" : L.role === "background" ? "#888" : "#e5c07b";
  row.innerHTML =
    `<span class="indent" style="width:${depth*16+8}px"></span>` +
    `<span class="arrow hidden">&#9654;</span>` +
    `<span class="icon" style="color:${iconColor}">${typeIcon(L.type)}</span>` +
    `<span class="node-name">${esc(L.textContent || L.name || L.id)}</span>` +
    `<span class="node-type">${L.type}</span>` +
    `<span class="vis-toggle ${S.visible.has(L.id)?'':'off'}">${S.visible.has(L.id)?'&#128065;':'&#128064;'}</span>`;

  row.querySelector(".vis-toggle").onclick = e => {
    e.stopPropagation();
    S.visible.has(L.id) ? S.visible.delete(L.id) : S.visible.add(L.id);
    renderTree(); renderStage();
  };
  row.onclick = () => { S.selected = S.selected === L.id ? null : L.id; renderTree(); renderStage(); renderInspector(); };
  parent.appendChild(row);
}

function groupAllVisible(g) { return g.childIds.every(id => S.visible.has(id)); }
function esc(s) { const d = document.createElement("span"); d.textContent = s; return d.innerHTML; }

/* ---- Stage ---- */
const stageEl = document.getElementById("stage");
function renderStage() {
  stageEl.innerHTML = "";
  if (S.view === "interactive") {
    const wrap = document.createElement("div");
    wrap.className = "stage-inner";
    wrap.style.width = DATA.canvas.width + "px";
    wrap.style.height = DATA.canvas.height + "px";
    wrap.style.maxWidth = "100%";
    if (DATA.background) wrap.style.background = DATA.background;
    [...DATA.layers].sort((a,b) => a.zIndex - b.zIndex).forEach(L => {
      if (!S.visible.has(L.id)) return;
      const img = document.createElement("img");
      img.className = "layer-img";
      img.src = L.src;
      wrap.appendChild(img);
    });
    // Highlight selected layer or group bbox.
    const sel = S.selected;
    let hlBox = null, hlLabel = "";
    if (sel && layerById[sel]) {
      hlBox = layerById[sel].bbox; hlLabel = layerById[sel].textContent || layerById[sel].id;
    } else if (sel && groupById[sel] && groupById[sel].bbox) {
      hlBox = groupById[sel].bbox; hlLabel = groupById[sel].name || sel;
    }
    if (hlBox) {
      const b = document.createElement("div");
      b.className = "bbox-hl";
      b.style.left = (hlBox.x/DATA.canvas.width*100)+"%";
      b.style.top = (hlBox.y/DATA.canvas.height*100)+"%";
      b.style.width = (hlBox.w/DATA.canvas.width*100)+"%";
      b.style.height = (hlBox.h/DATA.canvas.height*100)+"%";
      b.innerHTML = `<span class="bbox-label">${esc(hlLabel)}</span>`;
      wrap.appendChild(b);
    }
    stageEl.appendChild(wrap);
  } else {
    const src = DATA.images[S.view];
    if (!src) { stageEl.innerHTML = '<div class="stage-note">No image</div>'; return; }
    const img = document.createElement("img");
    img.src = src; img.style.maxWidth = "100%"; img.style.height = "auto";
    stageEl.appendChild(img);
  }
}

/* ---- Inspector ---- */
const inspEl = document.getElementById("inspector");
function renderInspector() {
  if (!S.selected) { inspEl.innerHTML = '<div class="inspector-empty">Select a layer</div>'; return; }
  const L = layerById[S.selected];
  const G = groupById[S.selected];
  if (G) { renderGroupInspector(G); return; }
  if (!L) { inspEl.innerHTML = '<div class="inspector-empty">Unknown</div>'; return; }
  let h = '<div class="prop-grid">';
  h += prop("ID", L.id) + prop("Type", L.type) + prop("Role", L.role);
  h += prop("Position", `${L.bbox.x}, ${L.bbox.y}`);
  h += prop("Size", `${L.bbox.w} × ${L.bbox.h}`);
  h += prop("z-index", L.zIndex);
  if (L.textContent) h += prop("Text", `"${L.textContent}"`);
  if (L.textColor) h += prop("Color", `<span class="color-chip" style="background:${L.textColor}"></span>${L.textColor}`);
  if (L.fontSize) h += prop("Font Size", Math.round(L.fontSize) + "px");
  if (L.fontWeight) h += prop("Weight", L.fontWeight);
  if (L.fill) {
    if (L.fill.type === "solid") h += prop("Fill", `<span class="color-chip" style="background:${L.fill.color}"></span>${L.fill.color}`);
    else h += prop("Fill", L.fill.type);
  }
  h += '</div>';
  h += `<img class="inspector-thumb" src="${L.src}" alt="${esc(L.id)}" />`;
  inspEl.innerHTML = h;
}
function renderGroupInspector(G) {
  let h = '<div class="prop-grid">';
  h += prop("Group", G.name || G.id) + prop("Children", G.childIds.length);
  if (G.bbox) {
    h += prop("Position", `${G.bbox.x}, ${G.bbox.y}`);
    h += prop("Size", `${G.bbox.w} × ${G.bbox.h}`);
  }
  h += '</div>';
  inspEl.innerHTML = h;
}
function prop(label, val) { return `<span class="label">${label}</span><span class="value">${val}</span>`; }

/* ---- Toolbar ---- */
document.getElementById("all-on").onclick = () => { DATA.layers.forEach(L => S.visible.add(L.id)); renderTree(); renderStage(); };
document.getElementById("all-off").onclick = () => { S.visible.clear(); renderTree(); renderStage(); };
document.getElementById("expand-all").onclick = () => { (DATA.groups||[]).forEach(g => S.expanded.add(g.id)); renderTree(); };
document.getElementById("collapse-all").onclick = () => { S.expanded.clear(); renderTree(); };

renderTree(); renderStage(); renderInspector();
</script>
</body>
</html>
"""
