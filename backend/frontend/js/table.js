// app/static/js/table.js
// STABLE VERSION — layout-safe + rules support

let LEAGUES = [];
let RULES_BY_LEAGUE = {};
let SEASONS = [];
let TEAM_LOGOS = {};

const state = {
  leagueId: "PL",
  seasonId: "2025-26",
};

let currentSheetType = null;

const $ = (sel) => document.querySelector(sel);

const els = {
  competitionLabel: $("#competitionLabel"),
  seasonLabel: $("#seasonLabel"),
  standingsTitle: $("#standingsTitle"),
  standingsBody: $("#standingsBody"),

  competitionFilter: $("#competitionFilter"),
  seasonFilter: $("#seasonFilter"),
  resetBtn: $("#resetFilters"),

  sheetBackdrop: $("#filterSheet"),
  sheet: $("#filterSheet .sheet"),
  sheetTitle: $("#sheetTitle"),
  sheetBody: $("#sheetBody"),
  sheetSaveBtn: $("#sheetSaveBtn"),
  sheetBackBtn: $("#sheetBackBtn"),
  sheetCloseBtn: $("#sheetCloseBtn"),
};

/* ===================== LOGO RESOLVER ===================== */
/**
 * Football-Data team names sometimes don't match your JSON keys exactly.
 * Example: "Manchester City FC" vs "Manchester City".
 *
 * This resolver:
 * 1) exact match
 * 2) normalized match (case-insensitive, remove accents/punctuation, remove FC/CF/etc)
 * 3) contains match (only if unambiguous)
 * 4) fallback default
 */
let TEAM_LOGOS_NORM = null;          // normalized index: normalizedName -> logoUrl
const LOGO_RESOLVE_CACHE = {};       // teamName -> resolvedUrl (per current league)

function normalizeTeamName(name) {
  return (name || "")
    .toLowerCase()
    .normalize("NFD")                 // split accents
    .replace(/[\u0300-\u036f]/g, "")  // remove accents
    .replace(/\b(fc|cf|ssc|ac|cd|ud|as|sv|afc|sc|ksc)\b/g, "") // remove common tokens
    .replace(/[^a-z0-9\s]/g, " ")     // punctuation -> space
    .replace(/\s+/g, " ")            // collapse spaces
    .trim();
}

function rebuildTeamLogoIndex() {
  TEAM_LOGOS_NORM = {};
  for (const [teamName, url] of Object.entries(TEAM_LOGOS)) {
    const key = normalizeTeamName(teamName);
    if (!key) continue;
    if (!TEAM_LOGOS_NORM[key]) TEAM_LOGOS_NORM[key] = url;
  }

  // Clear per-team cache whenever we rebuild (league changed)
  for (const k in LOGO_RESOLVE_CACHE) delete LOGO_RESOLVE_CACHE[k];
}

function resolveTeamLogo(teamName) {
  const fallback = "/static/icons/common/default.png";
  if (!teamName) return fallback;

  // Cached?
  if (LOGO_RESOLVE_CACHE[teamName]) return LOGO_RESOLVE_CACHE[teamName];

  // 1) exact match
  if (TEAM_LOGOS[teamName]) {
    LOGO_RESOLVE_CACHE[teamName] = TEAM_LOGOS[teamName];
    return TEAM_LOGOS[teamName];
  }

  // Ensure normalized index exists
  if (!TEAM_LOGOS_NORM) rebuildTeamLogoIndex();

  const norm = normalizeTeamName(teamName);

  // 2) normalized exact match
  if (TEAM_LOGOS_NORM[norm]) {
    LOGO_RESOLVE_CACHE[teamName] = TEAM_LOGOS_NORM[norm];
    return TEAM_LOGOS_NORM[norm];
  }

  // 3) contains match (only if single match to avoid wrong logos)
  const matches = [];
  for (const [k, url] of Object.entries(TEAM_LOGOS_NORM)) {
    if (!k) continue;
    if (k.includes(norm) || norm.includes(k)) matches.push(url);
    if (matches.length > 1) break; // ambiguous
  }
  if (matches.length === 1) {
    LOGO_RESOLVE_CACHE[teamName] = matches[0];
    return matches[0];
  }

  // 4) fallback
  LOGO_RESOLVE_CACHE[teamName] = fallback;
  return fallback;
}

/* ===================== META ===================== */
async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();

  if (!res.ok || !Array.isArray(data.leagues)) {
    throw new Error("Failed to load meta");
  }

  LEAGUES = data.leagues;
  RULES_BY_LEAGUE = {};

  LEAGUES.forEach(l => {
    RULES_BY_LEAGUE[l.id] = l.rules || {};
  });

  const preferred = LEAGUES.find(l => l.id === "PL") || LEAGUES[0];
  state.leagueId = preferred.id;

  setSeasonsForLeague(state.leagueId);

  if (!SEASONS.some(s => s.id === state.seasonId)) {
    state.seasonId = SEASONS[0]?.id;
  }
}

function setSeasonsForLeague(leagueId) {
  const league = LEAGUES.find(l => l.id === leagueId);
  SEASONS = (league?.seasons || []).map(s => ({
    id: s,
    label: s.replace("-", "/"),
  }));
}

/* ===================== TEAMS ===================== */
async function loadTeams() {
  const res = await fetch(`/api/teams?league=${state.leagueId}`);
  TEAM_LOGOS = res.ok ? await res.json() : {};

  // IMPORTANT: rebuild normalized index + clear logo cache whenever teams mapping changes
  rebuildTeamLogoIndex();
}

/* ===================== API ===================== */
async function fetchStandings() {
  const qs = new URLSearchParams({
    league: state.leagueId,
    season: state.seasonId,
  });

  const res = await fetch(`/api/standings?${qs}`);
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.error || "Failed to load standings");
  }

  if (!Array.isArray(data)) {
    throw new Error("Invalid standings format");
  }

  return data;
}

/* ===================== RULES → CSS CLASS ===================== */
function rankBand(pos) {
  const rules = RULES_BY_LEAGUE[state.leagueId] || {};

  if (Array.isArray(rules.ucl) && rules.ucl.includes(pos)) return "q-ucl";
  if (Array.isArray(rules.uel) && rules.uel.includes(pos)) return "q-uel";
  if (Array.isArray(rules.uecl) && rules.uecl.includes(pos)) return "q-conf";

  if (rules.relegation_playoff === pos) return "q-rel-playoff";
  if (Array.isArray(rules.relegation_auto) && rules.relegation_auto.includes(pos)) {
    return "q-rel";
  }

  return "";
}

/* ===================== RENDER ===================== */
function renderTopLabels() {
  const league = LEAGUES.find(l => l.id === state.leagueId);
  const season = SEASONS.find(s => s.id === state.seasonId);

  els.competitionLabel.textContent = league?.label || state.leagueId;
  els.seasonLabel.textContent = season?.label || state.seasonId;
  els.standingsTitle.textContent = league?.label || "Table";
}

function renderStandingsRows(rows) {
  els.standingsBody.innerHTML = rows.map((r, idx) => {
    const pos = idx + 1;
    const band = rankBand(pos);
    const logoUrl = resolveTeamLogo(r.team);

    return `
      <tr class="team-row ${band}">
        <td class="col-pos">
          <span class="pos-pill">${pos}</span>
        </td>

        <td class="col-team">
          <div class="team-cell">
            <img class="team-badge-sm"
                 src="${logoUrl}"
                 alt="">
            <span class="team-name">${escapeHtml(r.team)}</span>
          </div>
        </td>

        <td class="col-num">${r.pl}</td>
        <td class="col-num">${r.w}</td>
        <td class="col-num">${r.d}</td>
        <td class="col-num">${r.l}</td>
        <td class="col-num col-g">${r.gf}:${r.ga}</td>
        <td class="col-num">${r.gd}</td>
        <td class="col-num col-pts">${r.pts}</td>

        <td class="col-form">
          <div class="form">
            ${renderFormDots(r.form)}
          </div>
        </td>
      </tr>
    `;
  }).join("");
}

function renderFormDots(form) {
  return (form || []).slice(-5).map(v => {
    const cls = v === "W" ? "w" : v === "D" ? "d" : "l";
    return `<span class="form-dot ${cls}" title="${v}">${v}</span>`;
  }).join("");
}

function renderErrorRow(msg) {
  els.standingsBody.innerHTML = `
    <tr>
      <td colspan="10" style="padding:16px;color:red">
        ${escapeHtml(msg)}
      </td>
    </tr>
  `;
}

/* ===================== SHEETS ===================== */
function openSheet(type) {
  currentSheetType = type;

  els.sheetBackdrop.hidden = false;
  requestAnimationFrame(() => {
    els.sheetBackdrop.classList.add("open");
    els.sheet.classList.add("open");
  });

  els.sheetTitle.textContent = type === "competition" ? "Leagues" : "Seasons";
  els.sheetBody.innerHTML = "";

  const list = type === "competition" ? LEAGUES : SEASONS;

  list.forEach(i => {
    els.sheetBody.innerHTML += `
      <label class="sheet-option">
        <span>${escapeHtml(i.label)}</span>
        <input type="radio" name="${type}" value="${i.id}"
          ${i.id === (type === "competition" ? state.leagueId : state.seasonId) ? "checked" : ""}>
      </label>
    `;
  });

  document.body.style.overflow = "hidden";
}

function closeSheet() {
  els.sheetBackdrop.classList.remove("open");
  els.sheet.classList.remove("open");

  els.sheetBackdrop.hidden = true;
  els.sheetBody.innerHTML = "";
  currentSheetType = null;

  document.body.style.overflow = "";
}

async function applySheetSelection() {
  if (currentSheetType === "competition") {
    const checked = document.querySelector("input[name='competition']:checked");
    if (checked) {
      state.leagueId = checked.value;
      setSeasonsForLeague(state.leagueId);
      state.seasonId = SEASONS[0]?.id;
      await loadTeams();
    }
  }

  if (currentSheetType === "season") {
    const checked = document.querySelector("input[name='season']:checked");
    if (checked) state.seasonId = checked.value;
  }

  await refresh();
  closeSheet();
}

/* ===================== REFRESH ===================== */
async function refresh() {
  renderTopLabels();

  try {
    const rows = await fetchStandings();
    renderStandingsRows(rows);
  } catch (e) {
    renderErrorRow(e.message);
  }
}

/* ===================== UTILS ===================== */
function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, m =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m])
  );
}

/* ===================== INIT ===================== */
document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadMeta();
    await loadTeams();
    await refresh();
  } catch (e) {
    renderErrorRow("Initialization failed");
  }

  els.competitionFilter.onclick = () => openSheet("competition");
  els.seasonFilter.onclick = () => openSheet("season");

  els.sheetSaveBtn.onclick = applySheetSelection;
  els.sheetCloseBtn.onclick = closeSheet;
  els.sheetBackBtn.onclick = closeSheet;

  els.sheetBackdrop.onclick = (e) => {
    if (e.target === els.sheetBackdrop) closeSheet();
  };

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.sheetBackdrop.hidden) closeSheet();
  });
});
