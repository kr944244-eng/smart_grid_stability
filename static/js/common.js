/* ============ GridSense shared JS: theme, API, charts ============ */
function toggleTheme(){
  const cur = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", cur);
  try { localStorage.setItem("gs-theme", cur); } catch(e) {}
  setThemeIcon();
  window.dispatchEvent(new Event("themechange"));
}
function setThemeIcon(){
  const b = document.getElementById("themeBtn");
  if (b) b.textContent = document.documentElement.getAttribute("data-theme") === "light" ? "🌙" : "☀️";
}
document.addEventListener("DOMContentLoaded", setThemeIcon);

async function api(url, body){
  const opt = body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {};
  const r = await fetch(url, opt);
  if (r.status === 401){ location.href = "/login"; throw new Error("login required"); }
  return r.json();
}
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = t => String(t).replace(/[<>&]/g, c => ({"<":"&lt;", ">":"&gt;", "&":"&amp;"})[c]);

/* Real-world labels for each model input */
const PARAM_INFO = {
  tau: {label:"Reaction time", unit:"s"},
  p:   {label:"Power demand", unit:"p.u."},
  g:   {label:"Price sensitivity", unit:""},
};
const NODE_NAMES = {1:"Power Plant", 2:"Residential Area", 3:"Industrial Zone", 4:"Commercial Hub"};
const kindOf = f => f.replace(/\d/, "");
const nodeOf = f => +f.slice(-1);
const prettyParam = f => `${PARAM_INFO[kindOf(f)].label} · ${NODE_NAMES[nodeOf(f)]}`;

/* ---------- SVG charts (no internet / library needed) ---------- */
function frame(el, h){ return {w: Math.max(el.clientWidth, 280), h}; }

function drawSweep(el, xs, ys, cx, cy, xlabel){
  const {w,h} = frame(el, 300), m = {l:52, r:16, t:24, b:44}, C = css("--cyan"), A = css("--amber"), R = css("--red");
  const X = v => m.l + (v - xs[0]) / (xs[xs.length-1] - xs[0]) * (w - m.l - m.r);
  const Y = v => m.t + (1 - v/100) * (h - m.t - m.b);
  let g = `<defs><linearGradient id="ga" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${C}" stop-opacity=".35"/><stop offset="1" stop-color="${C}" stop-opacity="0"/></linearGradient></defs>`;
  [0,25,50,75,100].forEach(t => g += `<line class="ax" x1="${m.l}" x2="${w-m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="axt" x="${m.l-8}" y="${Y(t)+4}" text-anchor="end">${t}%</text>`);
  for (let i=0;i<=5;i++){ const v = xs[0] + i*(xs[xs.length-1]-xs[0])/5; g += `<text class="axt" x="${X(v)}" y="${h-m.b+18}" text-anchor="middle">${v.toFixed(2)}</text>`; }
  const line = xs.map((x,i) => `${i?"L":"M"}${X(x).toFixed(1)} ${Y(ys[i]).toFixed(1)}`).join(" ");
  g += `<path d="${line} L${X(xs[xs.length-1])} ${Y(0)} L${X(xs[0])} ${Y(0)} Z" fill="url(#ga)"/>`;
  g += `<line x1="${m.l}" x2="${w-m.r}" y1="${Y(50)}" y2="${Y(50)}" stroke="${R}" stroke-opacity=".75" stroke-dasharray="6 6"/>`;
  g += `<text x="${w-m.r}" y="${Y(50)-6}" fill="${R}" font-size="11" text-anchor="end">50% stability line</text>`;
  g += `<path d="${line}" fill="none" stroke="${C}" stroke-width="3" stroke-linejoin="round"/>`;
  g += `<line x1="${X(cx)}" x2="${X(cx)}" y1="${m.t}" y2="${Y(0)}" stroke="${A}" stroke-opacity=".5" stroke-dasharray="3 4"/>`;
  g += `<circle cx="${X(cx)}" cy="${Y(cy)}" r="7" fill="${A}" stroke="#fff" stroke-width="2"/>`;
  g += `<text x="${Math.min(X(cx)+10, w-m.r-100)}" y="${cy > 85 ? Y(cy)+22 : Math.max(Y(cy)-12, m.t+14)}" fill="${A}" font-size="12" font-weight="600">your grid: ${cy.toFixed(1)}%</text>`;
  g += `<text class="axl" x="${(m.l+w-m.r)/2}" y="${h-6}" text-anchor="middle">${esc(xlabel)}</text>`;
  g += `<text class="axl" transform="translate(13 ${(m.t+h-m.b)/2}) rotate(-90)" text-anchor="middle">Chance of stability</text>`;
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${g}</svg>`;
}

function drawGrouped(el, labels, series, lo, hi){
  const {w,h} = frame(el, 260), m = {l:44, r:8, t:30, b:40};
  const Y = v => m.t + (1 - (v-lo)/(hi-lo)) * (h - m.t - m.b);
  const gw = (w - m.l - m.r) / labels.length, bw = Math.min(30, gw/(series.length+1));
  let g = series.map((s,i) => `<rect x="${m.l + i*110}" y="4" width="10" height="10" rx="3" fill="${s.color}"/><text class="axl" x="${m.l + i*110 + 16}" y="13">${s.name}</text>`).join("");
  for (let t=lo; t<=hi; t+=10) g += `<line class="ax" x1="${m.l}" x2="${w-m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="axt" x="${m.l-6}" y="${Y(t)+4}" text-anchor="end">${t}%</text>`;
  labels.forEach((lab,j) => {
    const x0 = m.l + j*gw + (gw - bw*series.length - 4*(series.length-1))/2;
    series.forEach((s,i) => {
      const v = s.values[j], x = x0 + i*(bw+4);
      g += `<rect x="${x}" y="${Y(v)}" width="${bw}" height="${Y(lo)-Y(v)}" rx="6" fill="${s.color}"><title>${s.name}: ${v.toFixed(2)}%</title></rect>`;
      g += `<text class="axt" x="${x+bw/2}" y="${Y(v)-5}" text-anchor="middle" font-size="10">${v.toFixed(1)}</text>`;
    });
    const words = lab.split(" ");
    g += `<text class="axl" x="${m.l + j*gw + gw/2}" y="${h-m.b+16}" text-anchor="middle">${esc(words[0])}</text>`;
    if (words[1]) g += `<text class="axl" x="${m.l + j*gw + gw/2}" y="${h-m.b+30}" text-anchor="middle">${esc(words.slice(1).join(" "))}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${g}</svg>`;
}

function drawHBars(el, labels, values, colors){
  const {w,h} = frame(el, 300), m = {l:150, r:50, t:4, b:4};
  const max = Math.max(...values), rh = (h - m.t - m.b) / labels.length;
  let g = "";
  labels.forEach((lab,i) => {
    const y = m.t + i*rh, bwid = Math.max(2, Math.max(0, values[i])/max * (w - m.l - m.r));
    g += `<text class="axl" x="${m.l-8}" y="${y+rh/2+4}" text-anchor="end" font-size="11">${esc(lab)}</text>`;
    g += `<rect x="${m.l}" y="${y+3}" width="${w-m.l-m.r}" height="${rh-6}" rx="5" fill="${css("--line")}"/>`;
    g += `<rect x="${m.l}" y="${y+3}" width="${bwid}" height="${rh-6}" rx="5" fill="${colors[i]}"><title>${esc(lab)}: ${values[i]}</title></rect>`;
    g += `<text class="axt" x="${m.l+bwid+6}" y="${y+rh/2+4}">${values[i].toFixed(3)}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${g}</svg>`;
}

function shortParam(f){ return `${PARAM_INFO[kindOf(f)].label} (${NODE_NAMES[nodeOf(f)].split(" ")[0]})`; }
