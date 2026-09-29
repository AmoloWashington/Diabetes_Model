// Shared helpers: API access, formatting, toasts, safe Markdown, video capture.

export async function api(path, body, { method } = {}) {
  const opts = { method: method || (body === undefined ? "GET" : "POST"), headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the GlucoLab server.");
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    if (data && data.detail) {
      msg = Array.isArray(data.detail)
        ? data.detail.map((d) => `${(d.loc || []).slice(1).join(".") || "input"}: ${d.msg}`).join("; ")
        : String(data.detail);
    }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

let toastTimer;
export function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), isError ? 6000 : 3000);
}

export const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d);
export const pct = (v, d = 1) => (v === null || v === undefined ? "–" : (100 * v).toFixed(d) + "%");

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe Markdown: input is escaped first, then a small subset is rendered.
export function markdown(src) {
  const lines = esc(src).split("\n");
  let html = "", inList = null, inCode = false, code = [], table = [];
  const inline = (t) => t
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  const closeList = () => { if (inList) { html += `</${inList}>`; inList = null; } };
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\|?\s*:?-{2,}/.test(r));
    html += "<div class='table-wrap'><table>" + rows.map((r, i) => {
      const cells = r.replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const tag = i === 0 ? "th" : "td";
      return "<tr>" + cells.map((c) => `<${tag}>${c}</${tag}>`).join("") + "</tr>";
    }).join("") + "</table></div>";
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      if (inCode) { html += `<pre><code>${code.join("\n")}</code></pre>`; code = []; inCode = false; }
      else { closeList(); flushTable(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(raw); continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) { closeList(); table.push(line.trim()); continue; }
    flushTable();
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { closeList(); const n = Math.min(m[1].length + 2, 6); html += `<h${n}>${inline(m[2])}</h${n}>`; }
    else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (inList !== "ul") { closeList(); html += "<ul>"; inList = "ul"; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (inList !== "ol") { closeList(); html += "<ol>"; inList = "ol"; } html += `<li>${inline(m[1])}</li>`; }
    else if (line.trim() === "") { closeList(); }
    else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  if (inCode) html += `<pre><code>${code.join("\n")}</code></pre>`;
  closeList(); flushTable();
  return html;
}

// Record any canvas to a WebM video with MediaRecorder.
export function makeRecorder(getCanvas, button, filename) {
  let rec = null, chunks = [];
  const label = button.textContent;
  button.addEventListener("click", () => {
    if (rec) { rec.stop(); return; }
    const canvas = getCanvas();
    if (!canvas || !canvas.captureStream || typeof MediaRecorder === "undefined") {
      toast("Video recording is not supported in this browser.", true);
      return;
    }
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
    chunks = [];
    rec = new MediaRecorder(canvas.captureStream(30), mimeType ? { mimeType, videoBitsPerSecond: 6e6 } : undefined);
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${filename}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      rec = null;
      button.textContent = label;
      button.classList.remove("rec");
      toast("Video saved.");
    };
    rec.start(250);
    button.textContent = "■ Stop recording";
    button.classList.add("rec");
  });
}
