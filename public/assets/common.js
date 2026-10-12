// yes, if: shared helpers for the homepage and console (no dependencies).
(() => {
  const NS = "http://www.w3.org/2000/svg";

  async function api(path, options = {}) {
    const res = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      const err = new Error(detail || body.error || `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return body;
  }

  const money = (v) => "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const hour = (dt) => {
    const h = Math.floor(dt / 3600) % 24;
    return String(h).padStart(2, "0") + ":" + String(Math.floor((dt % 3600) / 60)).padStart(2, "0");
  };
  const actionLabel = (a) => ({ ALLOW: "Allow", STEP_UP: "Step up", HOLD: "Hold", BLOCK: "Block" })[a] || a;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceMax(v) {
    if (v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  // Line chart. series: [{name, color, values, dash, width}]; opts: {marker: {at: 0..1, label}, yFormat, xLabel}
  function lineChart(target, series, opts = {}) {
    const W = 640, H = opts.height || 300, L = 54, R = 14, T = 14, B = 34;
    const n = Math.max(...series.map((s) => s.values.length));
    const ymax = niceMax(Math.max(...series.flatMap((s) => s.values)));
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (v) => T + (1 - v / ymax) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": opts.label || "chart" });
    const fmt = opts.yFormat || ((v) => v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(v));
    for (let k = 0; k <= 4; k++) {
      const v = (ymax * k) / 4;
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "rgba(238,242,246,.07)" }, svg);
      el("text", { x: L - 8, y: y(v) + 3, "text-anchor": "end" }, svg).textContent = fmt(v);
    }
    if (opts.xLabel) el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = opts.xLabel;
    if (opts.marker) {
      const mx = L + opts.marker.at * (W - L - R);
      el("line", { x1: mx, x2: mx, y1: T, y2: H - B, stroke: "rgba(245,184,92,.7)", "stroke-dasharray": "4 4" }, svg);
      el("text", { x: mx + 6, y: T + 10, fill: "#f5b85c", style: "fill:#f5b85c" }, svg).textContent = opts.marker.label;
    }
    for (const s of series) {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }, svg);
    }
    target.replaceChildren(svg);
    if (opts.legend !== false) {
      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = series.map((s) => `<span><i style="background:${s.color}${s.dash ? ";opacity:.7" : ""}"></i>${esc(s.name)}</span>`).join("");
      target.appendChild(legend);
    }
  }

  // Grouped bars with an optional overlaid line. bars: [{label, value, color}], line: {values, color, max}
  function barChart(target, bars, opts = {}) {
    const W = 640, H = opts.height || 260, L = 46, R = opts.line ? 46 : 14, T = 14, B = 34;
    const ymax = opts.max || niceMax(Math.max(...bars.map((b) => b.value)));
    const slot = (W - L - R) / bars.length;
    const y = (v) => T + (1 - v / ymax) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": opts.label || "chart" });
    const fmt = opts.yFormat || ((v) => String(v));
    for (let k = 0; k <= 4; k++) {
      const v = (ymax * k) / 4;
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "rgba(238,242,246,.07)" }, svg);
      el("text", { x: L - 8, y: y(v) + 3, "text-anchor": "end" }, svg).textContent = fmt(v);
    }
    bars.forEach((b, i) => {
      const bw = Math.min(46, slot * 0.55), bx = L + i * slot + (slot - bw) / 2;
      el("rect", { x: bx, y: y(b.value), width: bw, height: Math.max(0, H - B - y(b.value)), rx: 6, fill: b.color || "#3fdcaa" }, svg);
      el("text", { x: bx + bw / 2, y: y(b.value) - 6, "text-anchor": "middle", style: "fill:#eef2f6" }, svg).textContent = fmt(b.value);
      el("text", { x: bx + bw / 2, y: H - 12, "text-anchor": "middle" }, svg).textContent = b.label;
    });
    if (opts.line) {
      const { values, color, max, min = 0, format } = opts.line;
      const ly = (v) => T + (1 - (v - min) / (max - min)) * (H - T - B);
      const d = values.map((v, i) => `${i ? "L" : "M"}${(L + i * slot + slot / 2).toFixed(1)},${ly(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: color, "stroke-width": 2 }, svg);
      values.forEach((v, i) => el("circle", { cx: L + i * slot + slot / 2, cy: ly(v), r: 3.5, fill: color }, svg));
      for (let k = 0; k <= 4; k++) {
        const v = min + ((max - min) * k) / 4;
        el("text", { x: W - R + 8, y: ly(v) + 3 }, svg).textContent = format ? format(v) : v.toFixed(2);
      }
    }
    target.replaceChildren(svg);
  }

  function reveal() {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const items = document.querySelectorAll(".reveal");
    if (reduced || !("IntersectionObserver" in window)) return items.forEach((e) => e.classList.add("in"));
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    items.forEach((e) => io.observe(e));
  }

  window.yesif = { api, money, pct, hour, actionLabel, esc, lineChart, barChart, reveal };
})();
