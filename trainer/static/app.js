/* OFC Pineapple regular trainer - frontend */
"use strict";

// ============ utilities ============

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const SUIT_CHAR = { s: "♠", h: "♥", d: "♦", c: "♣" };
const ROWS = ["top", "middle", "bottom"];
const ROW_CAP = { top: 3, middle: 5, bottom: 5 };
const ROW_JP = { top: "上", middle: "中", bottom: "下", discard: "捨" };

function rankDisp(r) {
  return r === "T" ? "10" : r;
}

function cardEl(card, opts = {}) {
  const el = document.createElement("div");
  const suit = card[1];
  el.className = `card ${suit}`;
  if (opts.small) el.classList.add("small");
  el.dataset.card = card;
  el.innerHTML = `<span>${rankDisp(card[0])}</span><span class="suit">${SUIT_CHAR[suit] || "?"}</span>`;
  for (const cls of opts.classes || []) el.classList.add(cls);
  if (opts.onClick) {
    el.classList.add("clickable");
    el.addEventListener("click", opts.onClick);
  }
  return el;
}

function slotEl(small) {
  const el = document.createElement("div");
  el.className = "cardslot";
  if (small) el.style.cssText = "width:34px;height:48px";
  return el;
}

function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return Number(x).toFixed(digits);
}

function fmtSigned(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  const v = Number(x);
  return (v > 0 ? "+" : "") + v.toFixed(digits);
}

function pct(x) {
  if (x === null || x === undefined) return "-";
  return (100 * x).toFixed(0) + "%";
}

function toast(message, cls = "", ms = 2200) {
  const wrap = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${cls}`;
  el.textContent = message;
  wrap.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// ============ accounts ============
// Local tool: an account is a named history bucket, not a login. The id rides
// on every request so the server scopes hands and mistakes to it.

const ACCOUNT_KEY = "ofc-trainer.account-id";

const Account = {
  id: Number(localStorage.getItem(ACCOUNT_KEY)) || null,
  list: [],

  async refresh() {
    const res = await api("/api/accounts");
    this.list = res.accounts;
    if (!this.list.some((a) => a.id === this.id)) {
      this.id = res.current;
      localStorage.setItem(ACCOUNT_KEY, String(this.id));
    }
    this.render();
  },

  render() {
    const sel = $("#account-select");
    sel.innerHTML = "";
    for (const acc of this.list) {
      const opt = document.createElement("option");
      opt.value = String(acc.id);
      opt.textContent = `${acc.name}（${acc.hands}ハンド）`;
      if (acc.id === this.id) opt.selected = true;
      sel.appendChild(opt);
    }
  },

  async select(id) {
    this.id = Number(id);
    localStorage.setItem(ACCOUNT_KEY, String(this.id));
    await this.refresh();
    await refreshStats();
    if ($("#view-mistakes").classList.contains("active")) Mistakes.refresh();
  },

  async create(name) {
    const acc = await api("/api/accounts", { method: "POST", body: { name } });
    await this.select(acc.id);
    toast(`アカウント「${acc.name}」に切り替えました`);
  },

  async remove(id) {
    await api(`/api/accounts/${id}`, { method: "DELETE" });
    this.id = null;
    localStorage.removeItem(ACCOUNT_KEY);
    await this.refresh();
    await refreshStats();
  },
};

// The quantity the candidate list is ordered by. Not always `ev`: at T0 the
// engine ranks on `candidate_score` and `ev` is not monotone in that order, so
// a delta taken on `ev` can come out positive for a worse-ranked line.
function rankScore(cand) {
  const m = (cand && cand.metrics) || {};
  return m.rank_score != null ? m.rank_score : m.ev;
}

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  if (Account.id) headers["X-Account-Id"] = String(Account.id);
  const res = await fetch(path, {
    headers,
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* noop */ }
    throw new Error(detail);
  }
  return res.json();
}

function actionText(action) {
  if (!action) return "-";
  const span = document.createElement("span");
  span.className = "action-str";
  const parts = [];
  const byRow = {};
  for (const [card, row] of action.placements || []) {
    (byRow[row] = byRow[row] || []).push(card);
  }
  for (const row of ROWS) {
    if (byRow[row]) {
      parts.push(`<span class="r-${row}">${ROW_JP[row]}:${byRow[row].map(c => rankDisp(c[0]) + SUIT_CHAR[c[1]]).join("")}</span>`);
    }
  }
  if (action.discard) {
    parts.push(`<span class="r-discard">${ROW_JP.discard}:${rankDisp(action.discard[0]) + SUIT_CHAR[action.discard[1]]}</span>`);
  }
  span.innerHTML = parts.join(" ");
  return span;
}

function renderStaticBoard(container, board, opts = {}) {
  container.innerHTML = "";
  const small = !!opts.small;
  for (const row of ROWS) {
    const rowEl = document.createElement("div");
    rowEl.className = "row";
    const cards = (board && board[row]) || [];
    for (const card of cards) {
      const extra = opts.highlight && opts.highlight.has(card) ? ["staged"] : [];
      rowEl.appendChild(cardEl(card, { small, classes: extra }));
    }
    for (let i = cards.length; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl(small));
    container.appendChild(rowEl);
  }
}

// ============ global stats ============

async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("#gs-hands").textContent = s.hands;
    $("#gs-decisions").textContent = s.decisions;
    $("#gs-evloss").textContent = fmt(s.total_ev_loss, 1);
    $("#gs-mistakes").textContent = s.mistakes;
  } catch (e) { /* stats are non-critical */ }
}

// ============ view switching ============

$$("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$("nav button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".view").forEach((v) => v.classList.remove("active"));
    $(`#view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "mistakes") Mistakes.refresh();
  });
});

// ============ Training ============

const Training = {
  sid: null,
  state: null,
  staged: [],            // [[card,row], ...] client-side placement
  selectedCard: null,
  pollTimer: null,
  lastGradedKey: null,   // avoid re-handling the same grading
  showBest: false,       // grade panel board toggle
  busy: false,

  options() {
    return {
      autoContinue: $("#opt-auto-continue").checked,
      forceBest: $("#opt-force-best").checked,
      skipObvious: $("#opt-skip-obvious").checked,
      autoNew: $("#opt-auto-new").checked,
      precision: $("#opt-precision").value,
      method: $("#opt-method").value,
      threshold: parseFloat($("#opt-threshold").value) || 1.0,
    };
  },

  async newHand() {
    if (this.busy) return;
    this.busy = true;
    this.staged = [];
    this.selectedCard = null;
    this.lastGradedKey = null;
    $("#grade-area").innerHTML = '<p class="muted">配置を確定すると、全候補のEVランキングと採点が表示されます。</p>';
    try {
      const res = await api("/api/training/new", {
        method: "POST",
        body: {
          position: "random",
          precision: this.options().precision,
          method: this.options().method,
          mistake_threshold: this.options().threshold,
        },
      });
      this.sid = res.session_id;
      this.state = res.state;
      this.render();
      this.startPolling();
    } catch (e) {
      toast("ハンド開始に失敗: " + e.message, "bad", 4000);
    } finally {
      this.busy = false;
    }
  },

  startPolling() {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = setInterval(() => this.poll(), 800);
  },

  async poll() {
    if (!this.sid || this.busy) return;
    const phase = this.state && this.state.phase;
    // poll while the server is working, or while we wait for eval (skip-obvious)
    const needPoll =
      phase === "dealing" || phase === "opp_turn" ||
      (phase === "hero_turn" && this.state.eval_status === "running");
    if (!needPoll) return;
    try {
      const res = await api(`/api/training/${this.sid}`);
      const prevPhase = phase;
      this.state = res;
      if (res.phase !== prevPhase || res.eval_status !== "running") {
        this.render();
      } else {
        this.renderStatus();
      }
      if (res.phase === "complete" && prevPhase !== "complete") {
        this.afterAdvance();
        refreshStats();
      }
      this.maybeAutoAct();
    } catch (e) { /* transient poll errors are fine */ }
  },

  maybeAutoAct() {
    const st = this.state;
    if (!st || st.phase !== "hero_turn") return;
    if (this.options().skipObvious && st.eval_status === "ready" && st.candidate_count === 1 && !this.busy) {
      this.playBest(true);
    }
  },

  requiredCount() {
    return this.state && this.state.street === 0 ? 5 : 2;
  },

  stagedCards() {
    return this.staged.map(([c]) => c);
  },

  async act(payload) {
    if (this.busy) return;
    this.busy = true;
    this.renderStatus("採点中…");
    let acted = false;
    try {
      const res = await api(`/api/training/${this.sid}/act`, { method: "POST", body: payload });
      this.state = res;
      this.staged = [];
      this.selectedCard = null;
      this.showBest = false;
      this.render();
      acted = true;
    } catch (e) {
      toast("エラー: " + e.message, "bad", 4000);
      this.render();
    } finally {
      this.busy = false;
    }
    if (acted) await this.handleGrading();
  },

  check() {
    const need = this.requiredCount();
    if (this.staged.length !== need) return;
    const placements = this.staged.map(([c, r]) => [c, r]);
    let discard = null;
    if (this.state.street > 0) {
      const used = new Set(this.stagedCards());
      discard = (this.state.dealt || []).find((c) => !used.has(c)) || null;
    }
    this.act({ placements, discard });
  },

  playBest(silent = false) {
    if (!silent) toast("最善手をプレイ", "", 1200);
    this.act({ auto_best: true });
  },

  async continueGame(use) {
    if (this.busy || !this.sid) return;
    this.busy = true;
    try {
      const res = await api(`/api/training/${this.sid}/continue`, { method: "POST", body: { use } });
      this.state = res;
      this.render();
      this.afterAdvance();
    } catch (e) {
      toast("エラー: " + e.message, "bad", 4000);
    } finally {
      this.busy = false;
    }
  },

  async handleGrading() {
    const st = this.state;
    if (st.phase !== "graded" || !st.grading) return;
    const g = st.grading;
    const key = `${st.hand_no}-${g.street}`;
    if (this.lastGradedKey === key) return;
    this.lastGradedKey = key;
    refreshStats();

    const opts = this.options();
    if (opts.forceBest) {
      await this.continueGame("best");
      return;
    }
    if (opts.autoContinue && g.is_best) {
      toast("✓ 最善手！", "good", 1400);
      await this.continueGame("user");
      return;
    }
    if (g.mistake) toast(`ミス記録: EVロス ${fmt(g.ev_loss)}`, "bad", 2500);
  },

  afterAdvance() {
    const st = this.state;
    if (st.phase === "complete") {
      this.showResult();
      refreshStats();
      if (this.options().autoNew) {
        setTimeout(() => { this.closeModal(); this.newHand(); }, 2600);
      }
    }
  },

  // ---------- rendering ----------

  render() {
    const st = this.state;
    if (!st) return;
    // labels
    $("#tr-table-title").textContent = `テーブル — ハンド #${st.hand_no || 1} / ストリート T${st.street}`;
    $("#hero-pos-label").textContent = st.position === "first" ? "先行" : "後攻";
    $("#opp-pos-label").textContent = st.position === "first" ? "後攻" : "先行";
    $("#hero-fl-label").style.display = st.hero_fl ? "" : "none";
    $("#opp-fl-label").style.display = st.opp_fl ? "" : "none";

    renderStaticBoard($("#opp-board"), st.opp_board, {});
    this.renderHeroBoard();
    this.renderTray();
    this.renderStatus();
    this.renderButtons();
    if (st.phase === "graded" && st.grading) this.renderGrading(st.grading);
  },

  renderHeroBoard() {
    const st = this.state;
    const container = $("#hero-board");
    container.innerHTML = "";
    const stagedByRow = {};
    let overlay = null;
    if (st.phase === "graded" && st.grading) {
      const src = this.showBest ? st.grading.best : st.grading.user;
      overlay = { placements: src.action.placements || [], best: this.showBest };
    }
    for (const row of ROWS) {
      const rowEl = document.createElement("div");
      rowEl.className = "row";
      const committed = (st.hero_board && st.hero_board[row]) || [];
      for (const card of committed) rowEl.appendChild(cardEl(card));

      let extras = [];
      if (overlay) {
        extras = overlay.placements.filter(([, r]) => r === row).map(([c]) => c);
        for (const card of extras) {
          rowEl.appendChild(cardEl(card, { classes: overlay.best ? ["ghost"] : ["staged"] }));
        }
      } else if (st.phase === "hero_turn") {
        extras = this.staged.filter(([, r]) => r === row).map(([c]) => c);
        for (const card of extras) {
          rowEl.appendChild(
            cardEl(card, {
              classes: ["staged"],
              onClick: (ev) => { ev.stopPropagation(); this.unstage(card); },
            })
          );
        }
        const free = ROW_CAP[row] - committed.length - extras.length;
        if (free > 0 && st.phase === "hero_turn") {
          rowEl.classList.add("droppable");
          rowEl.addEventListener("click", () => this.placeTo(row));
        }
      }
      const total = committed.length + extras.length;
      for (let i = total; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl());
      container.appendChild(rowEl);
    }
  },

  renderTray() {
    const st = this.state;
    const tray = $("#dealt-tray");
    tray.innerHTML = '<span class="tray-label">配られたカード</span>';
    if (st.phase !== "hero_turn" && st.phase !== "graded") return;
    const stagedSet = new Set(this.stagedCards());
    const dealt = st.dealt || [];
    let gradedDiscard = null;
    if (st.phase === "graded" && st.grading) {
      const src = this.showBest ? st.grading.best : st.grading.user;
      gradedDiscard = src.action.discard;
      const placed = new Set((src.action.placements || []).map(([c]) => c));
      for (const card of dealt) {
        if (placed.has(card)) continue;
        const classes = card === gradedDiscard ? ["discard-mark", "dim"] : ["dim"];
        tray.appendChild(cardEl(card, { classes }));
      }
      return;
    }
    for (const card of dealt) {
      if (stagedSet.has(card)) continue;
      const classes = [];
      if (card === this.selectedCard) classes.push("selected");
      // auto-discard preview: T1+ with 2 staged, the leftover is the discard
      if (st.street > 0 && this.staged.length === 2) classes.push("discard-mark");
      tray.appendChild(
        cardEl(card, {
          classes,
          onClick: () => this.selectCard(card),
        })
      );
    }
  },

  renderStatus(msgOverride) {
    const st = this.state;
    const line = $("#status-line");
    if (msgOverride) {
      line.innerHTML = `<div class="spinner"></div> ${msgOverride}`;
      return;
    }
    if (!st) { line.textContent = "「新しいハンド」で開始します。"; return; }
    switch (st.phase) {
      case "dealing":
        line.innerHTML = '<div class="spinner"></div> 配っています…';
        break;
      case "opp_turn":
        line.innerHTML = '<div class="spinner"></div> 相手が考えています…';
        break;
      case "hero_turn": {
        const need = this.requiredCount();
        const evalNote =
          st.eval_status === "running" ? '<span class="muted">（裏でEV解析中…）</span>' :
          st.eval_status === "ready" ? '<span class="muted">（解析済み）</span>' :
          st.eval_status === "error" ? '<span style="color:var(--bad)">（解析エラー）</span>' : "";
        line.innerHTML = `あなたの番です。${st.street === 0 ? "5枚全て配置" : "2枚配置（残り1枚は自動で捨て札）"}してください (${this.staged.length}/${need}) ${evalNote}`;
        break;
      }
      case "graded":
        line.textContent = "採点結果を確認して、続行してください。";
        break;
      case "complete":
        line.textContent = "ハンド終了。";
        break;
      case "error":
        line.innerHTML = `<span style="color:var(--bad)">エラー: ${st.error || "不明"}</span>`;
        break;
      default:
        line.textContent = "";
    }
  },

  renderButtons() {
    const st = this.state;
    const heroTurn = st && st.phase === "hero_turn";
    $("#btn-check").disabled = !(heroTurn && this.staged.length === this.requiredCount()) || this.busy;
    $("#btn-clear").disabled = !(heroTurn && this.staged.length > 0);
    $("#btn-play-best").disabled = !heroTurn || this.busy;
  },

  renderGrading(g) {
    const area = $("#grade-area");
    area.innerHTML = "";

    const badge = document.createElement("div");
    const lossClass = g.is_best ? "best" : g.ev_loss < (this.options().threshold || 1.0) ? "ok" : "blunder";
    badge.className = `grade-badge ${lossClass}`;
    badge.textContent = g.is_best
      ? "✓ 最善手！"
      : `EVロス ${fmt(g.ev_loss)}（${g.rank ? g.rank + "位" : "圏外"} / ${g.candidate_count}候補）`;
    area.appendChild(badge);

    // view toggle
    const toggleWrap = document.createElement("div");
    toggleWrap.style.cssText = "margin:10px 0;display:flex;gap:8px;align-items:center";
    const mkToggle = (label, isBest) => {
      const b = document.createElement("button");
      b.className = "small" + (this.showBest === isBest ? " primary" : "");
      b.textContent = label;
      b.addEventListener("click", () => { this.showBest = isBest; this.render(); });
      return b;
    };
    toggleWrap.append("盤面表示:", mkToggle("自分の手", false), mkToggle("最善手", true));
    area.appendChild(toggleWrap);

    // candidates table
    const table = document.createElement("table");
    table.className = "cands";
    table.innerHTML = `<thead><tr>
      <th>#</th><th style="text-align:left">アクション</th><th>EV差</th><th>バースト</th><th>FL</th>
    </tr></thead>`;
    const tbody = document.createElement("tbody");
    const userKey = g.user_key;
    const cands = g.candidates || [];
    // Absolute EV is only comparable inside one position, so the ranking reads
    // better as a loss against the best line than as raw scores.
    const bestEv = g.best ? rankScore(g.best) : (cands[0] && rankScore(cands[0]));
    cands.forEach((cand, i) => {
      const tr = document.createElement("tr");
      if (cand.key === userKey) tr.classList.add("user-row");
      if (i === 0) tr.classList.add("best-row");
      tr.innerHTML = `<td>${i + 1}</td>`;
      const tdA = document.createElement("td");
      tdA.appendChild(actionText(cand.action));
      if (cand.key === userKey) tdA.append(" ← あなた");
      tr.appendChild(tdA);
      const delta = rankScore(cand) - bestEv;
      const cell = i === 0 || delta > -0.005
        ? `<td class="ev-best">best</td>`
        : `<td class="ev-neg">${fmt(delta)}</td>`;
      tr.innerHTML += `${cell}
        <td>${pct(cand.metrics.bust_rate)}</td>
        <td>${pct(cand.metrics.fl_rate)}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    area.appendChild(table);

    if (g.evaluator) {
      const note = document.createElement("p");
      note.className = "muted";
      note.style.marginTop = "6px";
      note.textContent = `評価: ${g.evaluator}`;
      area.appendChild(note);
    }

    const btns = document.createElement("div");
    btns.className = "controls";
    const bUser = document.createElement("button");
    bUser.textContent = "このまま進む";
    bUser.addEventListener("click", () => this.continueGame("user"));
    const bBest = document.createElement("button");
    bBest.className = "primary";
    bBest.textContent = "最善手で進む";
    bBest.addEventListener("click", () => this.continueGame("best"));
    btns.append(bUser, bBest);
    area.appendChild(btns);
  },

  showResult() {
    const st = this.state;
    const r = st.result;
    if (!r) return;
    const modal = $("#modal");
    const scoreCls = r.score > 0 ? "ev-pos" : r.score < 0 ? "ev-neg" : "";
    const flLine = (fl, who) =>
      fl && fl.entry ? `<div class="k">${who} FL</div><div>突入（${(fl.type || "").toUpperCase()}・${fl.cards}枚 / EV +${fmt(fl.ev, 1)}）</div>` : "";
    modal.innerHTML = `
      <h3>ハンド結果</h3>
      <div class="result-score ${scoreCls}">${fmtSigned(r.score, 0)} pt</div>
      <div class="result-grid">
        <div class="k">ライン</div><div>${r.line_text}</div>
        <div class="k">ロイヤリティ</div><div>あなた ${r.hero_royalty} / 相手 ${r.opp_royalty}</div>
        ${r.hero_busted ? '<div class="k">バースト</div><div style="color:var(--bad)">あなたはバースト</div>' : ""}
        ${r.opp_busted ? '<div class="k">バースト</div><div style="color:var(--good)">相手がバースト</div>' : ""}
        ${r.scoop ? '<div class="k">スクープ</div><div>+3</div>' : ""}
        ${flLine(r.hero_fl, "あなた")}
        ${flLine(r.opp_fl, "相手")}
        <div class="k">このハンドのEVロス</div><div>${fmt(r.total_ev_loss)}（ミス ${r.mistakes} 回）</div>
      </div>
      <div class="controls">
        <button class="primary" id="modal-next">新しいハンド</button>
        <button id="modal-close">閉じる</button>
      </div>`;
    $("#modal-backdrop").classList.add("open");
    $("#modal-next").addEventListener("click", () => { this.closeModal(); this.newHand(); });
    $("#modal-close").addEventListener("click", () => this.closeModal());
  },

  closeModal() {
    $("#modal-backdrop").classList.remove("open");
  },

  // ---------- placement interactions ----------

  selectCard(card) {
    this.selectedCard = this.selectedCard === card ? null : card;
    this.renderTray();
  },

  placeTo(row) {
    const st = this.state;
    if (!st || st.phase !== "hero_turn") return;
    if (this.staged.length >= this.requiredCount()) return;
    let card = this.selectedCard;
    if (!card) {
      // convenience: no selection -> take first remaining dealt card
      const stagedSet = new Set(this.stagedCards());
      card = (st.dealt || []).find((c) => !stagedSet.has(c));
      if (!card) return;
    }
    const committed = (st.hero_board[row] || []).length;
    const stagedInRow = this.staged.filter(([, r]) => r === row).length;
    if (committed + stagedInRow >= ROW_CAP[row]) return;
    this.staged.push([card, row]);
    this.selectedCard = null;
    this.render();
  },

  unstage(card) {
    this.staged = this.staged.filter(([c]) => c !== card);
    this.render();
  },

  clearStaged() {
    this.staged = [];
    this.selectedCard = null;
    this.render();
  },
};

$("#btn-new-hand").addEventListener("click", () => Training.newHand());
$("#btn-check").addEventListener("click", () => Training.check());
$("#btn-clear").addEventListener("click", () => Training.clearStaged());
$("#btn-play-best").addEventListener("click", () => Training.playBest());

// ============ Editor ============

const Editor = {
  zone: "dealt",
  zones: {
    "hero-top": [], "hero-middle": [], "hero-bottom": [],
    "opp-top": [], "opp-middle": [], "opp-bottom": [],
    dealt: [], dead: [],
  },
  result: null,
  previewIdx: null,

  allCards() {
    const cards = [];
    for (const s of ["s", "h", "d", "c"]) {
      for (const r of ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]) {
        cards.push(r + s);
      }
    }
    return cards;
  },

  usedSet() {
    const used = new Set();
    for (const z of Object.values(this.zones)) for (const c of z) used.add(c);
    return used;
  },

  zoneCap(zone) {
    if (zone.endsWith("top")) return 3;
    if (zone.endsWith("middle") || zone.endsWith("bottom")) return 5;
    if (zone === "dealt") return 5;
    return 52;
  },

  toggleCard(card) {
    // if assigned anywhere, remove; else add to active zone
    for (const [name, arr] of Object.entries(this.zones)) {
      const idx = arr.indexOf(card);
      if (idx >= 0) { arr.splice(idx, 1); this.render(); return; }
    }
    const arr = this.zones[this.zone];
    if (arr.length >= this.zoneCap(this.zone)) { toast("この枠は満杯です", "warn"); return; }
    arr.push(card);
    this.previewIdx = null;
    this.result = null;
    this.render();
  },

  inferTurn() {
    const heroCount = this.zones["hero-top"].length + this.zones["hero-middle"].length + this.zones["hero-bottom"].length;
    if (heroCount === 0) return 0;
    if (heroCount >= 5 && (heroCount - 5) % 2 === 0) return Math.min(4, 1 + (heroCount - 5) / 2);
    return null;
  },

  heroBoard() {
    return { top: this.zones["hero-top"], middle: this.zones["hero-middle"], bottom: this.zones["hero-bottom"] };
  },
  oppBoard() {
    return { top: this.zones["opp-top"], middle: this.zones["opp-middle"], bottom: this.zones["opp-bottom"] };
  },

  async evaluate() {
    const turnSel = $("#ed-turn").value;
    const turn = turnSel === "auto" ? this.inferTurn() : parseInt(turnSel, 10);
    const dealt = this.zones.dealt;
    const need = turn === 0 ? 5 : 3;
    if (turn === null) { toast("盤面の枚数が不正です（5+2n枚にしてください）", "warn", 3000); return; }
    if (dealt.length !== need) { toast(`T${turn} は配札 ${need} 枚が必要です`, "warn", 3000); return; }
    $("#ed-status").innerHTML = '<div class="spinner" style="display:inline-block;vertical-align:-2px"></div> 評価中…（精度によっては数十秒かかります）';
    $("#ed-evaluate").disabled = true;
    try {
      const res = await api("/api/editor/evaluate", {
        method: "POST",
        body: {
          hero_board: this.heroBoard(),
          opp_board: this.oppBoard(),
          dealt,
          dead: this.zones.dead,
          turn,
          position: $("#ed-position").value,
          precision: $("#ed-precision").value,
          method: $("#ed-method").value,
        },
      });
      this.result = res;
      this.previewIdx = null;
      $("#ed-status").textContent = `T${res.turn} / ${res.candidate_count} 候補 / ${res.evaluator} / ${fmt(res.elapsed, 1)}秒`;
      this.renderResults();
    } catch (e) {
      $("#ed-status").innerHTML = `<span style="color:var(--bad)">評価エラー: ${e.message}</span>`;
    } finally {
      $("#ed-evaluate").disabled = false;
    }
  },

  randomDeal() {
    const turnSel = $("#ed-turn").value;
    const turn = turnSel === "auto" ? (this.inferTurn() ?? 0) : parseInt(turnSel, 10);
    const need = turn === 0 ? 5 : 3;
    const used = this.usedSet();
    for (const c of this.zones.dealt) used.delete(c);
    const pool = this.allCards().filter((c) => !used.has(c));
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    this.zones.dealt = pool.slice(0, need);
    this.result = null;
    this.render();
  },

  clearAll() {
    for (const key of Object.keys(this.zones)) this.zones[key] = [];
    this.result = null;
    this.previewIdx = null;
    this.render();
    $("#ed-results").innerHTML = '<p class="muted">ポジションを作って「評価する」を押してください。</p>';
    $("#ed-status").textContent = "";
  },

  render() {
    // picker
    const picker = $("#ed-picker");
    picker.innerHTML = "";
    const used = this.usedSet();
    for (const card of this.allCards()) {
      const el = cardEl(card, { onClick: () => this.toggleCard(card) });
      if (used.has(card)) el.classList.add("used");
      picker.appendChild(el);
    }
    // boards with preview overlay
    let previewPlacements = [];
    let previewDiscard = null;
    if (this.result && this.previewIdx !== null) {
      const cand = this.result.candidates[this.previewIdx];
      if (cand) { previewPlacements = cand.action.placements || []; previewDiscard = cand.action.discard; }
    }
    const heroC = $("#ed-hero-board");
    heroC.innerHTML = "";
    for (const row of ROWS) {
      const rowEl = document.createElement("div");
      rowEl.className = "row";
      for (const card of this.zones[`hero-${row}`]) {
        rowEl.appendChild(cardEl(card, { small: true, onClick: () => this.toggleCard(card) }));
      }
      const ghosts = previewPlacements.filter(([, r]) => r === row);
      for (const [card] of ghosts) rowEl.appendChild(cardEl(card, { small: true, classes: ["ghost"] }));
      const total = this.zones[`hero-${row}`].length + ghosts.length;
      for (let i = total; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl(true));
      heroC.appendChild(rowEl);
    }
    renderStaticBoard($("#ed-opp-board"), this.oppBoard(), { small: true });
    // re-add click handlers for opp board cards (renderStaticBoard is static; simplest: rebuild)
    const oppC = $("#ed-opp-board");
    oppC.innerHTML = "";
    for (const row of ROWS) {
      const rowEl = document.createElement("div");
      rowEl.className = "row";
      for (const card of this.zones[`opp-${row}`]) {
        rowEl.appendChild(cardEl(card, { small: true, onClick: () => this.toggleCard(card) }));
      }
      for (let i = this.zones[`opp-${row}`].length; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl(true));
      oppC.appendChild(rowEl);
    }
    // trays
    const dealtTray = $("#ed-dealt-tray");
    dealtTray.innerHTML = '<span class="tray-label">配札</span>';
    for (const card of this.zones.dealt) {
      const classes = card === previewDiscard ? ["discard-mark"] : [];
      dealtTray.appendChild(cardEl(card, { small: true, classes, onClick: () => this.toggleCard(card) }));
    }
    const deadTray = $("#ed-dead-tray");
    deadTray.innerHTML = '<span class="tray-label">死札（自分の捨て札など）</span>';
    for (const card of this.zones.dead) {
      deadTray.appendChild(cardEl(card, { small: true, onClick: () => this.toggleCard(card) }));
    }
    if (this.result) this.renderResults();
  },

  renderResults() {
    const area = $("#ed-results");
    area.innerHTML = "";
    const res = this.result;
    if (!res || !res.candidates || !res.candidates.length) {
      area.innerHTML = '<p class="muted">候補がありません。</p>';
      return;
    }
    const best = res.candidates[0];
    const table = document.createElement("table");
    table.className = "cands";
    table.innerHTML = `<thead><tr>
      <th>#</th><th style="text-align:left">アクション</th><th>EV</th><th>ΔEV</th><th>バースト</th><th>FL</th>
    </tr></thead>`;
    const tbody = document.createElement("tbody");
    res.candidates.forEach((cand, i) => {
      const tr = document.createElement("tr");
      tr.className = "selectable";
      if (i === 0) tr.classList.add("best-row");
      if (i === this.previewIdx) tr.classList.add("preview-row");
      const delta = rankScore(cand) - rankScore(best);
      tr.innerHTML = `<td>${i + 1}</td>`;
      const tdA = document.createElement("td");
      tdA.appendChild(actionText(cand.action));
      tr.appendChild(tdA);
      tr.innerHTML += `
        <td class="${cand.metrics.ev >= 0 ? "ev-pos" : "ev-neg"}">${fmtSigned(cand.metrics.ev)}</td>
        <td class="${delta < -0.005 ? "ev-neg" : "muted"}">${fmt(delta)}</td>
        <td>${pct(cand.metrics.bust_rate)}</td>
        <td>${pct(cand.metrics.fl_rate)}</td>`;
      tr.addEventListener("click", () => {
        this.previewIdx = this.previewIdx === i ? null : i;
        this.render();
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    area.appendChild(table);
  },
};

$$("#ed-zones button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$("#ed-zones button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    Editor.zone = btn.dataset.zone;
  });
});
$("#ed-evaluate").addEventListener("click", () => Editor.evaluate());
$("#ed-clear").addEventListener("click", () => Editor.clearAll());
$("#ed-random").addEventListener("click", () => Editor.randomDeal());

// ============ Mistakes ============

const Mistakes = {
  items: [],
  current: null,   // detail object
  staged: [],
  selectedCard: null,

  async refresh() {
    try {
      const res = await api("/api/mistakes");
      this.items = res.mistakes || [];
      this.renderList();
    } catch (e) {
      toast("ミス一覧の取得に失敗: " + e.message, "bad");
    }
  },

  renderList() {
    const list = $("#mi-list");
    list.innerHTML = "";
    if (!this.items.length) {
      list.innerHTML = '<p class="muted">まだミスは記録されていません。トレーニングでプレイしましょう。</p>';
      return;
    }
    for (const item of this.items) {
      const el = document.createElement("div");
      el.className = "mistake-item";
      if (this.current && this.current.id === item.id) el.classList.add("active");
      const date = new Date(item.created_at * 1000);
      const dateStr = `${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
      el.innerHTML = `
        <span>T${item.street}</span>
        <span class="meta">${item.position === "first" ? "先行" : "後攻"} · ${dateStr}</span>
        ${item.solved ? '<span class="solved-mark">✓ 克服</span>' : item.retried ? `<span class="meta">再挑戦${item.retried}回</span>` : ""}
        <span class="loss">-${fmt(item.ev_loss)}</span>`;
      el.addEventListener("click", () => this.open(item.id));
      list.appendChild(el);
    }
  },

  async open(id) {
    try {
      this.current = await api(`/api/mistakes/${id}`);
      this.staged = [];
      this.selectedCard = null;
      this.renderDetail();
      this.renderList();
    } catch (e) {
      toast("読み込みエラー: " + e.message, "bad");
    }
  },

  requiredCount() {
    return this.current && this.current.street === 0 ? 5 : 2;
  },

  renderDetail(graded = null) {
    const m = this.current;
    const area = $("#mi-detail");
    area.innerHTML = "";
    if (!m) return;

    const info = document.createElement("p");
    info.className = "muted";
    info.style.marginBottom = "8px";
    info.textContent = `T${m.street} / ${m.position === "first" ? "先行" : "後攻"} / 当時のEVロス ${fmt(m.ev_loss)}`;
    area.appendChild(info);

    const oppLabel = document.createElement("div");
    oppLabel.className = "board-label";
    oppLabel.innerHTML = '<span class="tag">相手</span>';
    area.appendChild(oppLabel);
    const oppBoard = document.createElement("div");
    renderStaticBoard(oppBoard, m.opp_board, { small: true });
    area.appendChild(oppBoard);

    const heroLabel = document.createElement("div");
    heroLabel.className = "board-label";
    heroLabel.innerHTML = '<span class="tag">あなた</span>';
    area.appendChild(heroLabel);

    const heroBoard = document.createElement("div");
    for (const row of ROWS) {
      const rowEl = document.createElement("div");
      rowEl.className = "row";
      const committed = m.hero_board[row] || [];
      for (const card of committed) rowEl.appendChild(cardEl(card, { small: true }));
      const extras = this.staged.filter(([, r]) => r === row).map(([c]) => c);
      for (const card of extras) {
        rowEl.appendChild(cardEl(card, {
          small: true, classes: ["staged"],
          onClick: (ev) => { ev.stopPropagation(); this.staged = this.staged.filter(([c2]) => c2 !== card); this.renderDetail(); },
        }));
      }
      const total = committed.length + extras.length;
      if (!graded && total < ROW_CAP[row]) {
        rowEl.classList.add("droppable");
        rowEl.addEventListener("click", () => this.placeTo(row));
      }
      for (let i = total; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl(true));
      heroBoard.appendChild(rowEl);
    }
    area.appendChild(heroBoard);

    const tray = document.createElement("div");
    tray.className = "tray";
    tray.innerHTML = '<span class="tray-label">配られたカード</span>';
    const stagedSet = new Set(this.staged.map(([c]) => c));
    for (const card of m.dealt) {
      if (stagedSet.has(card)) continue;
      const classes = [];
      if (card === this.selectedCard) classes.push("selected");
      tray.appendChild(cardEl(card, {
        small: true, classes,
        onClick: () => { this.selectedCard = this.selectedCard === card ? null : card; this.renderDetail(); },
      }));
    }
    area.appendChild(tray);

    if (!graded) {
      const btns = document.createElement("div");
      btns.className = "controls";
      const bCheck = document.createElement("button");
      bCheck.className = "primary";
      bCheck.textContent = "この配置で答え合わせ";
      bCheck.disabled = this.staged.length !== this.requiredCount();
      bCheck.addEventListener("click", () => this.grade());
      const bClear = document.createElement("button");
      bClear.textContent = "クリア";
      bClear.addEventListener("click", () => { this.staged = []; this.renderDetail(); });
      const bDelete = document.createElement("button");
      bDelete.className = "danger";
      bDelete.textContent = "このミスを削除";
      bDelete.addEventListener("click", () => this.remove());
      btns.append(bCheck, bClear, bDelete);
      area.appendChild(btns);
      return;
    }

    // ---- graded view ----
    const badge = document.createElement("div");
    badge.className = `grade-badge ${graded.isBest ? "best" : graded.loss < 1.0 ? "ok" : "blunder"}`;
    badge.style.marginTop = "10px";
    badge.textContent = graded.found
      ? graded.isBest ? "✓ 最善手！克服しました" : `EVロス ${fmt(graded.loss)}（${graded.rank}位）`
      : "候補に見つかりません（当時の評価対象外の手）";
    area.appendChild(badge);

    const table = document.createElement("table");
    table.className = "cands";
    table.style.marginTop = "10px";
    table.innerHTML = "<thead><tr><th>#</th><th style='text-align:left'>アクション</th><th>EV差</th><th>バースト</th><th>FL</th></tr></thead>";
    const tbody = document.createElement("tbody");
    const rows = (m.candidates || []).slice(0, 10);
    const bestEv = rows[0] && rankScore(rows[0]);
    rows.forEach((cand, i) => {
      const tr = document.createElement("tr");
      if (i === 0) tr.classList.add("best-row");
      if (graded.found && i === graded.rank - 1) tr.classList.add("user-row");
      tr.innerHTML = `<td>${i + 1}</td>`;
      const tdA = document.createElement("td");
      tdA.appendChild(actionText(cand.action));
      tr.appendChild(tdA);
      const delta = rankScore(cand) - bestEv;
      const cell = i === 0 || delta > -0.005
        ? `<td class="ev-best">best</td>`
        : `<td class="ev-neg">${fmt(delta)}</td>`;
      tr.innerHTML += `${cell}<td>${pct(cand.metrics.bust_rate)}</td><td>${pct(cand.metrics.fl_rate)}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    area.appendChild(table);

    const btns = document.createElement("div");
    btns.className = "controls";
    const bRetry = document.createElement("button");
    bRetry.textContent = "もう一度挑戦";
    bRetry.addEventListener("click", () => { this.staged = []; this.renderDetail(); });
    btns.append(bRetry);
    area.appendChild(btns);
  },

  placeTo(row) {
    const m = this.current;
    if (!m || this.staged.length >= this.requiredCount()) return;
    let card = this.selectedCard;
    if (!card) {
      const stagedSet = new Set(this.staged.map(([c]) => c));
      card = m.dealt.find((c) => !stagedSet.has(c));
      if (!card) return;
    }
    const committed = (m.hero_board[row] || []).length;
    const stagedInRow = this.staged.filter(([, r]) => r === row).length;
    if (committed + stagedInRow >= ROW_CAP[row]) return;
    this.staged.push([card, row]);
    this.selectedCard = null;
    this.renderDetail();
  },

  async grade() {
    const m = this.current;
    const key = this.boardKey(this.resultingBoard());
    let found = false, rank = 0, loss = null, isBest = false;
    (m.candidates || []).forEach((cand, i) => {
      if (this.boardKey(cand.board) === key) {
        found = true;
        rank = i + 1;
        loss = rankScore(m.candidates[0]) - rankScore(cand);
        isBest = i === 0;
      }
    });
    this.renderDetail({ found, rank, loss, isBest });
    try {
      await api(`/api/mistakes/${m.id}/retry`, { method: "POST", body: { solved: !!isBest } });
      const item = this.items.find((x) => x.id === m.id);
      if (item) { item.retried = (item.retried || 0) + 1; item.solved = isBest ? 1 : 0; }
      this.renderList();
    } catch (e) { /* non-critical */ }
  },

  resultingBoard() {
    const m = this.current;
    const rows = { top: [...m.hero_board.top], middle: [...m.hero_board.middle], bottom: [...m.hero_board.bottom] };
    for (const [card, row] of this.staged) rows[row].push(card);
    return rows;
  },

  boardKey(board) {
    return ROWS.map((r) => [...(board[r] || [])].sort().join(",")).join("|");
  },

  async remove() {
    if (!this.current) return;
    try {
      await api(`/api/mistakes/${this.current.id}`, { method: "DELETE" });
      this.current = null;
      $("#mi-detail").innerHTML = '<p class="muted">左のリストからミスを選ぶと、同じ場面をもう一度プレイできます。</p>';
      this.refresh();
      refreshStats();
    } catch (e) {
      toast("削除に失敗: " + e.message, "bad");
    }
  },
};

// Precision is a particle count. Under the model there are no particles, so the
// control is disabled in these two views as it is in the hand-log one.
function syncMethodControls() {
  for (const [method, precision] of [
    ["#opt-method", "#opt-precision"],
    ["#ed-method", "#ed-precision"],
  ]) {
    const sel = $(method);
    if (sel) $(precision).disabled = sel.value === "model";
  }
}
$("#opt-method").addEventListener("change", syncMethodControls);
$("#ed-method").addEventListener("change", syncMethodControls);
syncMethodControls();

$("#mi-refresh").addEventListener("click", () => Mistakes.refresh());

// ============ account controls ============

$("#account-select").addEventListener("change", (ev) => {
  Account.select(ev.target.value).catch((e) => toast("切り替えに失敗: " + e.message, "bad"));
});

$("#account-new").addEventListener("click", async () => {
  const name = prompt("新しいアカウント名");
  if (!name || !name.trim()) return;
  try {
    await Account.create(name.trim());
  } catch (e) {
    toast("作成に失敗: " + e.message, "bad");
  }
});

$("#account-delete").addEventListener("click", async () => {
  const acc = Account.list.find((a) => a.id === Account.id);
  if (!acc) return;
  const msg = `「${acc.name}」の履歴（${acc.hands}ハンド / ${acc.mistakes}ミス）を完全に削除します。よろしいですか？`;
  if (!confirm(msg)) return;
  try {
    await Account.remove(acc.id);
    toast("削除しました");
  } catch (e) {
    toast("削除に失敗: " + e.message, "bad");
  }
});

// ============ init ============

Editor.render();
Account.refresh()
  .catch(() => { /* accounts are non-critical for the editor */ })
  .finally(() => refreshStats());
