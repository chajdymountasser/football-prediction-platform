// static/js/matches.js
// Matches page: meta → teams → fixtures → render
// + filter sheets (league/season/matchweek/clubs)
// + predict modal using /api/predict (random for now)
// ML-ready: /api/predict receives {league, season, matchday, matchId}

let LEAGUES = [];
let SEASONS = [];
let TEAM_LOGOS = {};
let CLUBS = [];

const state = {
  leagueId: "PL",
  seasonId: "2025-26",
  matchday: null, // null = let backend choose current MW
  selectedClubs: new Set(),
};

let currentSheetType = null;

// prediction
let fixturesIndex = new Map();     // matchId(string) -> match object
let selectedMatchId = null;
const predictionCache = new Map(); // key = league:season:matchId

const $ = (sel) => document.querySelector(sel);

const els = {
  competitionLabel: $("#competitionLabel"),
  seasonLabel: $("#seasonLabel"),
  matchweekLabel: $("#matchweekLabel"),
  clubsLabel: $("#clubsLabel"),

  matchweekTitle: $("#matchweekTitle"),
  matchweekDates: $("#matchweekDates"),

  fixturesContainer: $("#fixturesContainer"),

  competitionFilter: $("#competitionFilter"),
  seasonFilter: $("#seasonFilter"),
  matchweekFilter: $("#matchweekFilter"),
  clubsFilter: $("#clubsFilter"),
  resetBtn: $("#resetFilters"),

  prevWeekBtn: $("#prevWeekBtn"),
  nextWeekBtn: $("#nextWeekBtn"),

  // sheet
  sheetBackdrop: $("#filterSheet"),
  sheet: $("#filterSheet .sheet"),
  sheetTitle: $("#sheetTitle"),
  sheetBody: $("#sheetBody"),
  sheetSaveBtn: $("#sheetSaveBtn"),
  sheetBackBtn: $("#sheetBackBtn"),
  sheetCloseBtn: $("#sheetCloseBtn"),

  // predict modal
  predictBackdrop: $("#predictBackdrop"),
  predictModal: $("#predictModal"),
  predictCloseBtn: $("#predictCloseBtn"),
  predictRunBtn: $("#predictRunBtn"),

  pHomeBadge: $("#pHomeBadge"),
  pAwayBadge: $("#pAwayBadge"),
  pHomeName: $("#pHomeName"),
  pAwayName: $("#pAwayName"),
  pHomeMeta: $("#pHomeMeta"),
  pAwayMeta: $("#pAwayMeta"),
  pStatus: $("#pStatus"),
  pTimeOrScore: $("#pTimeOrScore"),

  pResultNote: $("#pResultNote"),
  pBars: $("#pBars"),
  barHomeLabel: $("#barHomeLabel"),
  barAwayLabel: $("#barAwayLabel"),
  barHomeFill: $("#barHomeFill"),
  barDrawFill: $("#barDrawFill"),
  barAwayFill: $("#barAwayFill"),
  barHomePct: $("#barHomePct"),
  barDrawPct: $("#barDrawPct"),
  barAwayPct: $("#barAwayPct"),
  pWinner: $("#pWinner"),
};

function predKey(matchId) {
  return `${state.leagueId}:${state.seasonId}:${String(matchId)}`;
}

/* ===================== META ===================== */
async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  if (!res.ok || !Array.isArray(data.leagues)) throw new Error("Failed to load meta");

  LEAGUES = data.leagues;

  const preferred = LEAGUES.find(l => l.id === "PL") || LEAGUES[0];
  state.leagueId = preferred?.id || "PL";

  setSeasonsForLeague(state.leagueId);

  if (!SEASONS.some(s => s.id === state.seasonId)) {
    state.seasonId = SEASONS[0]?.id || state.seasonId;
  }
}

function setSeasonsForLeague(leagueId) {
  const league = LEAGUES.find(l => l.id === leagueId);
  SEASONS = (league?.seasons || []).map(s => ({
    id: s,
    label: s.replace("-", "/"),
  }));
}

/* ===================== TEAMS / CLUBS ===================== */
async function loadTeams() {
  const res = await fetch(`/api/teams?league=${encodeURIComponent(state.leagueId)}`);
  TEAM_LOGOS = res.ok ? await res.json() : {};
  CLUBS = Object.keys(TEAM_LOGOS).sort((a, b) => a.localeCompare(b));
}

/* ===================== FETCH FIXTURES ===================== */
async function fetchFixtures() {
  const qs = new URLSearchParams({
    league: state.leagueId,
    season: state.seasonId,
  });

  if (state.matchday != null) qs.set("matchday", String(state.matchday));

  if (state.selectedClubs.size > 0) {
    qs.set("clubs", Array.from(state.selectedClubs).join(","));
  }

  const res = await fetch(`/api/fixtures?${qs.toString()}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Failed to load fixtures");

  if (typeof data.matchday === "number") state.matchday = data.matchday;
  else if (state.matchday == null) state.matchday = 1;

  const groups = data.fixtures || [];
  fixturesIndex = buildFixturesIndex(groups);
  setMatchweekDatesFromGroups(groups);
  return groups;
}

function buildFixturesIndex(groups) {
  const map = new Map();
  for (const g of groups) {
    for (const m of (g.matches || [])) {
      // IMPORTANT: store group date on each match so prediction can send it
      m.date = g.groupLabel; // "YYYY-MM-DD"
      map.set(String(m.id), m);
    }
  }
  return map;
}

function setMatchweekDatesFromGroups(groups) {
  const dates = (groups || []).map(g => g.groupLabel).filter(Boolean).sort();
  if (dates.length === 0) els.matchweekDates.textContent = "No dates";
  else if (dates.length === 1) els.matchweekDates.textContent = dates[0];
  else els.matchweekDates.textContent = `${dates[0]} → ${dates[dates.length - 1]}`;
}

/* ===================== RENDER ===================== */
function renderTopLabels() {
  const league = LEAGUES.find(l => l.id === state.leagueId);
  const season = SEASONS.find(s => s.id === state.seasonId);

  els.competitionLabel.textContent = league?.label || state.leagueId;
  els.seasonLabel.textContent = season?.label || state.seasonId;

  els.matchweekLabel.textContent = `MW${state.matchday ?? "?"}`;
  els.matchweekTitle.textContent = `Matchweek ${state.matchday ?? "?"}`;
  els.matchweekDates.textContent = "Loading…";

  if (state.selectedClubs.size === 0) els.clubsLabel.textContent = "Clubs";
  else if (state.selectedClubs.size === 1) els.clubsLabel.textContent = Array.from(state.selectedClubs)[0];
  else els.clubsLabel.textContent = `${state.selectedClubs.size} clubs`;
}

function renderFixtures(groups) {
  if (!els.fixturesContainer) return;

  if (!groups || groups.length === 0) {
    els.fixturesContainer.innerHTML = `<div class="fixtures-empty">No matches found for these filters.</div>`;
    return;
  }

  els.fixturesContainer.innerHTML = groups.map(renderWeekBlock).join("");

  els.fixturesContainer.querySelectorAll("[data-action='predict']").forEach(btn => {
    btn.addEventListener("click", () => openPredict(String(btn.dataset.matchId)));
  });
}

function renderWeekBlock(group) {
  const rows = (group.matches || []).map(renderMatchRow).join("");
  return `
    <section class="week-block">
      <div class="week-date">${escapeHtml(group.groupLabel)}</div>
      ${rows}
    </section>
  `;
}

function renderMatchRow(m) {
  const matchId = String(m.id);
  const isFinished = m.status === "FT";

  const center = isFinished && m.score
    ? `
      <div class="match-score-pill is-score">
        <span class="score-pill__home">${escapeHtml(m.score.home)}</span>
        <span class="score-pill__dash">-</span>
        <span class="score-pill__away">${escapeHtml(m.score.away)}</span>
      </div>`
    : `
      <div class="match-score-pill is-time">
        <span>${escapeHtml(m.kickoff ?? "--:--")}</span>
      </div>`;

  const conf = m.ml?.topLabel
    ? `<span class="confidence-pill" title="Prediction">${escapeHtml(m.ml.topLabel)}</span>`
    : "";

  const predictBtn = isFinished
    ? ""
    : `<button class="predict-btn" type="button" data-action="predict" data-match-id="${escapeHtml(matchId)}">Predict</button>`;

  return `
    <div class="match-row" data-match-id="${escapeHtml(matchId)}">
      <div class="match-status">${escapeHtml(m.status)}</div>

      <div class="match-teams">
        <div class="team team-home">
          <span class="team-name">${escapeHtml(m.home.name)}</span>
          <img class="team-badge" src="${escapeHtml(teamBadgeUrl(m.home.name, m.home.badge))}" alt="">
        </div>

        ${center}

        <div class="team team-away">
          <img class="team-badge" src="${escapeHtml(teamBadgeUrl(m.away.name, m.away.badge))}" alt="">
          <span class="team-name">${escapeHtml(m.away.name)}</span>
        </div>
      </div>

      <div class="match-actions">
        ${conf}
        ${predictBtn}
      </div>
    </div>
  `;
}

function teamBadgeUrl(teamName, fallbackFromApi) {
  return TEAM_LOGOS[teamName] || fallbackFromApi || "/static/icons/common/default.png";
}

/* ===================== SHEET ===================== */
function openSheet(type) {
  currentSheetType = type;

  els.sheetBackdrop.hidden = false;
  requestAnimationFrame(() => {
    els.sheetBackdrop.classList.add("open");
    els.sheet.classList.add("open");
  });

  if (type === "competition") {
    els.sheetTitle.textContent = "Leagues";
    renderRadioOptions("competition", LEAGUES.map(l => ({ id: l.id, label: l.label })), state.leagueId);
  } else if (type === "season") {
    els.sheetTitle.textContent = "Seasons";
    renderRadioOptions("season", SEASONS, state.seasonId);
  } else if (type === "matchweek") {
    els.sheetTitle.textContent = "Matchweeks";
    renderRadioOptions("matchweek", buildMatchweeksForLeague(), String(state.matchday ?? 1));
  } else if (type === "clubs") {
    els.sheetTitle.textContent = "Clubs";
    renderClubOptions();
  }

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

function renderRadioOptions(name, items, selectedId) {
  els.sheetBody.innerHTML = "";
  items.forEach(i => {
    const row = document.createElement("label");
    row.className = "sheet-option";
    row.innerHTML = `
      <span class="sheet-option-label">${escapeHtml(i.label)}</span>
      <input type="radio" name="${escapeHtml(name)}" value="${escapeHtml(i.id)}" ${String(i.id) === String(selectedId) ? "checked" : ""}>
    `;
    els.sheetBody.appendChild(row);
  });
}

function renderClubOptions() {
  els.sheetBody.innerHTML = "";
  CLUBS.forEach(club => {
    const row = document.createElement("label");
    row.className = "sheet-option";
    row.innerHTML = `
      <span class="sheet-option-label">${escapeHtml(club)}</span>
      <input type="checkbox" name="club" value="${escapeHtml(club)}" ${state.selectedClubs.has(club) ? "checked" : ""}>
    `;
    els.sheetBody.appendChild(row);
  });
}

async function applySheetSelection() {
  if (!currentSheetType) return;

  if (currentSheetType === "competition") {
    const checked = els.sheetBody.querySelector("input[name='competition']:checked");
    if (checked) {
      state.leagueId = checked.value;
      setSeasonsForLeague(state.leagueId);
      state.seasonId = SEASONS[0]?.id;
      state.matchday = null;
      state.selectedClubs.clear();
      predictionCache.clear();
      await loadTeams();
    }
  }

  if (currentSheetType === "season") {
    const checked = els.sheetBody.querySelector("input[name='season']:checked");
    if (checked) {
      state.seasonId = checked.value;
      state.matchday = null;
      predictionCache.clear();
    }
  }

  if (currentSheetType === "matchweek") {
    const checked = els.sheetBody.querySelector("input[name='matchweek']:checked");
    if (checked) state.matchday = Number(checked.value);
  }

  if (currentSheetType === "clubs") {
    const inputs = els.sheetBody.querySelectorAll("input[name='club']");
    state.selectedClubs.clear();
    inputs.forEach(input => input.checked && state.selectedClubs.add(input.value));
  }

  closeSheet();
  await refresh();
}

function buildMatchweeksForLeague() {
  return Array.from({ length: 38 }, (_, i) => ({ id: String(i + 1), label: `MW${i + 1}` }));
}

/* ===================== PREDICT MODAL ===================== */
function resetPredictUI() {
  els.pBars.hidden = true;
  els.pWinner.hidden = true;

  els.pResultNote.innerHTML = `Click <b>Run prediction</b> to generate Win/Draw/Lose probabilities.`;
  els.pWinner.textContent = "";

  els.barHomeLabel.textContent = "Home win";
  els.barAwayLabel.textContent = "Away win";

  els.barHomeFill.style.width = "0%";
  els.barDrawFill.style.width = "0%";
  els.barAwayFill.style.width = "0%";

  els.barHomePct.textContent = "0%";
  els.barDrawPct.textContent = "0%";
  els.barAwayPct.textContent = "0%";
}

function openPredict(matchId) {
  selectedMatchId = String(matchId);
  const m = fixturesIndex.get(selectedMatchId);
  if (!m) return;

  resetPredictUI();

  els.pHomeName.textContent = m.home.name;
  els.pAwayName.textContent = m.away.name;

  els.pHomeBadge.src = teamBadgeUrl(m.home.name, m.home.badge);
  els.pAwayBadge.src = teamBadgeUrl(m.away.name, m.away.badge);

  els.pHomeMeta.textContent = "Home";
  els.pAwayMeta.textContent = "Away";

  els.pStatus.textContent = m.status;
  els.pTimeOrScore.textContent =
    (m.status === "FT" && m.score) ? `${m.score.home} - ${m.score.away}` : (m.kickoff ?? "--:--");

  const k = predKey(selectedMatchId);
  if (predictionCache.has(k)) applyPredictionToUIAndRow(m, predictionCache.get(k));

  els.predictBackdrop.hidden = false;
  requestAnimationFrame(() => {
    els.predictBackdrop.classList.add("open");
    els.predictModal.classList.add("open");
  });

  document.body.style.overflow = "hidden";
}

function closePredict() {
  els.predictBackdrop.classList.remove("open");
  els.predictModal.classList.remove("open");
  els.predictBackdrop.hidden = true;
  document.body.style.overflow = "";
  selectedMatchId = null;
}

async function runPrediction(matchId) {
  const id = String(matchId);
  const m = fixturesIndex.get(id);
  if (!m) return;

  const k = predKey(id);
  if (predictionCache.has(k)) {
    applyPredictionToUIAndRow(m, predictionCache.get(k));
    return;
  }

  resetPredictUI();
  els.pResultNote.textContent = "Running model…";

    const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      league: state.leagueId,
      date: m.date,                 // already "YYYY-MM-DD" in your fixtures
      homeTeam: m.home.name,
      awayTeam: m.away.name
    })
  });


  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    els.pResultNote.textContent = data?.error || "Prediction failed.";
    return;
  }

  const probs = {
    homeWin: data.homeWin,
    draw: data.draw,
    awayWin: data.awayWin,
  };

  predictionCache.set(k, probs);
  applyPredictionToUIAndRow(m, probs);
}

function applyPredictionToUIAndRow(m, probs) {
  const topLabel = topOutcomeLabel(m, probs);
  m.ml = { ...probs, topLabel };

  els.pBars.hidden = false;

  els.barHomeLabel.textContent = `${m.home.name} win`;
  els.barAwayLabel.textContent = `${m.away.name} win`;

  setBar(els.barHomeFill, els.barHomePct, probs.homeWin);
  setBar(els.barDrawFill, els.barDrawPct, probs.draw);
  setBar(els.barAwayFill, els.barAwayPct, probs.awayWin);

  els.pWinner.hidden = false;
  els.pWinner.textContent = `Most likely: ${topLabel}`;
  els.pResultNote.textContent = "Prediction ready.";

  updateRowConfidencePill(m);
}

function updateRowConfidencePill(m) {
  const row = els.fixturesContainer?.querySelector(`[data-match-id="${CSS.escape(String(m.id))}"]`);
  if (!row) return;

  const actions = row.querySelector(".match-actions");
  if (!actions) return;

  let pill = actions.querySelector(".confidence-pill");
  if (!pill) {
    pill = document.createElement("span");
    pill.className = "confidence-pill";
    actions.insertBefore(pill, actions.firstChild);
  }
  pill.textContent = m.ml?.topLabel ?? "";
}

function topOutcomeLabel(m, probs) {
  const h = Number(probs.homeWin) || 0;
  const d = Number(probs.draw) || 0;
  const a = Number(probs.awayWin) || 0;

  const entries = [
    { p: h, label: `${m.home.name} ${Math.round(h * 100)}%` },
    { p: d, label: `Draw ${Math.round(d * 100)}%` },
    { p: a, label: `${m.away.name} ${Math.round(a * 100)}%` },
  ];
  entries.sort((x, y) => y.p - x.p);
  return entries[0].label;
}

function setBar(fillEl, pctEl, value01) {
  const v = clamp01(Number(value01));
  const pct = Math.round(v * 100);
  fillEl.style.width = `${pct}%`;
  pctEl.textContent = `${pct}%`;
}

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

/* ===================== REFRESH ===================== */
async function refresh() {
  renderTopLabels();
  try {
    const groups = await fetchFixtures();
    renderTopLabels();
    renderFixtures(groups);
  } catch (e) {
    console.error(e);
    els.fixturesContainer.innerHTML = `<div class="fixtures-empty">${escapeHtml(e.message)}</div>`;
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
    console.error(e);
  }

  els.competitionFilter?.addEventListener("click", () => openSheet("competition"));
  els.seasonFilter?.addEventListener("click", () => openSheet("season"));
  els.matchweekFilter?.addEventListener("click", () => openSheet("matchweek"));
  els.clubsFilter?.addEventListener("click", () => openSheet("clubs"));

  els.resetBtn?.addEventListener("click", async () => {
    const preferred = LEAGUES.find(l => l.id === "PL") || LEAGUES[0];
    state.leagueId = preferred?.id || "PL";
    setSeasonsForLeague(state.leagueId);
    state.seasonId = SEASONS[0]?.id || "2025-26";
    state.matchday = null;
    state.selectedClubs.clear();
    predictionCache.clear();
    await loadTeams();
    await refresh();
  });

  els.prevWeekBtn?.addEventListener("click", async () => {
    if (state.matchday == null) return;
    if (state.matchday > 1) {
      state.matchday--;
      await refresh();
    }
  });

  els.nextWeekBtn?.addEventListener("click", async () => {
    if (state.matchday == null) return;
    if (state.matchday < 38) {
      state.matchday++;
      await refresh();
    }
  });

  els.sheetSaveBtn?.addEventListener("click", applySheetSelection);
  els.sheetCloseBtn?.addEventListener("click", closeSheet);
  els.sheetBackBtn?.addEventListener("click", closeSheet);

  els.sheetBackdrop?.addEventListener("click", (e) => {
    if (e.target === els.sheetBackdrop) closeSheet();
  });

  els.predictCloseBtn?.addEventListener("click", closePredict);
  els.predictBackdrop?.addEventListener("click", (e) => {
    if (e.target === els.predictBackdrop) closePredict();
  });

  els.predictRunBtn?.addEventListener("click", () => {
    if (!selectedMatchId) return;
    runPrediction(selectedMatchId);
  });

  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (els.predictBackdrop && !els.predictBackdrop.hidden) closePredict();
    else if (els.sheetBackdrop && !els.sheetBackdrop.hidden) closeSheet();
  });
});
