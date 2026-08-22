/* Hand-history entry: transcribe a hand that already happened, then grade it.
 *
 * Loaded after app.js and reuses its globals ($, cardEl, api, toast, ...).
 *
 * The entry flow is a fixed ten-step walk -- five streets x two seats, in deal
 * order -- because that is the order the user reads a hand off a video in.  It
 * also means the app, not the user, decides whose board a card lands on and
 * which street it belongs to, which is where hand-typed records go wrong.
 *
 * The two seats are entered differently because they are seen differently.
 * For the hero all three dealt cards are typed and the leftover becomes the
 * discard automatically.  For the opponent only what they placed is visible,
 * so their steps take two cards, not three; their discards are an optional
 * eleventh step at the end, for hands where the muck was shown afterwards.
 */
"use strict";

const HL_SUITS = ["s", "h", "d", "c"];
const HL_RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const HL_DEAL = [5, 3, 3, 3, 3];
const HL_PLACE = [5, 2, 2, 2, 2];
const FL_CARDS = 14; // dealt to a Fantasyland seat
const FL_PLACE = 13; // set from them, in one action

function hlEmptySide() {
  return { dealt: [], placements: [], discard: null };
}

function hlEmptyStreets() {
  return [0, 1, 2, 3, 4].map((turn) => ({ turn, hero: hlEmptySide(), opp: hlEmptySide() }));
}

const HandLog = {
  heroPosition: "first",
  // Which seats are in Fantasyland. A seat in FL has no streets at all: it took
  // fourteen cards before the hand and set thirteen of them in one action.
  fl: { hero: false, opp: false },
  flHands: { hero: hlEmptySide(), opp: hlEmptySide() },
  streets: hlEmptyStreets(),
  step: 0, // index into stepList(); one past the end is the discard panel
  editingId: null,
  selected: null, // tray card awaiting a destination
  discardSlot: null, // opponent turn whose discard the next card fills
  discardOrder: [], // opponent turns in the order their discards were typed
  keyBuffer: "",
  list: [],
  checked: new Set(),
  poll: null,

  // ---------------- model ----------------

  seatOrder() {
    return this.heroPosition === "first" ? ["hero", "opp"] : ["opp", "hero"];
  },

  patternKey() {
    if (this.fl.hero && this.fl.opp) return "both";
    if (this.fl.hero) return "hero";
    if (this.fl.opp) return "opp";
    return "normal";
  },

  setPattern(key) {
    this.fl = {
      hero: key === "hero" || key === "both",
      opp: key === "opp" || key === "both",
    };
    this.reset(true);
  },

  /** The ordered list of entry steps for this hand's shape.
   *
   * Not a formula over the index any more: with a seat in Fantasyland the walk
   * is a different length and a different shape.  A Fantasyland seat
   * contributes one thirteen-card step and no streets; a normal seat
   * contributes its five.  Fantasyland comes first because that is when it is
   * dealt -- before the streets start.
   */
  stepList() {
    const steps = [];
    for (const seat of ["hero", "opp"]) {
      if (this.fl[seat]) steps.push({ kind: "fl", seat, turn: null });
    }
    if (!(this.fl.hero && this.fl.opp)) {
      for (let turn = 0; turn < 5; turn++) {
        for (const seat of this.seatOrder()) {
          if (!this.fl[seat]) steps.push({ kind: "street", seat, turn });
        }
      }
    }
    return steps.map((s, index) => ({
      ...s,
      index,
      position: s.seat === "hero" ? this.heroPosition : this.heroPosition === "first" ? "second" : "first",
    }));
  },

  stepInfo(index = this.step) {
    const steps = this.stepList();
    return index < steps.length ? steps[index] : null;
  },

  side(info) {
    return info.kind === "fl" ? this.flHands[info.seat] : this.streets[info.turn][info.seat];
  },

  /** How many cards this seat types in on this step.
   *
   * The asymmetry is the same one the streets have, for the same reason: the
   * hero knows all fourteen cards Fantasyland dealt them, while of the opponent
   * only the thirteen they set are ever visible.
   */
  dealCount(info) {
    if (info.kind === "fl") return info.seat === "hero" ? FL_CARDS : FL_PLACE;
    return info.seat === "hero" ? HL_DEAL[info.turn] : HL_PLACE[info.turn];
  },

  /** How many cards this seat places on this step. */
  placeCount(info) {
    return info.kind === "fl" ? FL_PLACE : HL_PLACE[info.turn];
  },

  /** True once every placement is in and only the optional discards remain. */
  inDiscardPhase() {
    return this.step >= this.stepList().length;
  },

  /** Cards entered anywhere in the hand -- the picker greys these out.
   *
   * Unions all three of dealt, placements and discard rather than trusting
   * `dealt` alone: the opponent's `dealt` is derived, so a set built from it
   * would leak a card back into the picker the moment the two disagreed.
   */
  used() {
    const set = new Set();
    const eat = (side) => {
      for (const card of side.dealt) set.add(card);
      for (const [card] of side.placements) set.add(card);
      if (side.discard) set.add(side.discard);
    };
    for (const street of this.streets) {
      for (const seat of ["hero", "opp"]) eat(street[seat]);
    }
    for (const seat of ["hero", "opp"]) eat(this.flHands[seat]);
    return set;
  },

  /** Board as entered so far, including the step in progress. */
  board(seat) {
    const rows = { top: [], middle: [], bottom: [] };
    if (this.fl[seat]) {
      for (const [card, row] of this.flHands[seat].placements) rows[row].push(card);
      return rows;
    }
    for (const street of this.streets) {
      for (const [card, row] of street[seat].placements) rows[row].push(card);
    }
    return rows;
  },

  discards(seat) {
    const out = this.streets.map((s) => s[seat].discard).filter(Boolean);
    if (this.fl[seat] && this.flHands[seat].discard) out.push(this.flHands[seat].discard);
    return out;
  },

  /** Cards dealt this step that have not been assigned a destination yet. */
  pending(info) {
    const side = this.side(info);
    const placed = new Set(side.placements.map(([c]) => c));
    if (side.discard) placed.add(side.discard);
    return side.dealt.filter((c) => !placed.has(c));
  },

  phase(info) {
    if (!info) return "discards";
    const side = this.side(info);
    return side.dealt.length < this.dealCount(info) ? "deal" : "place";
  },

  /** The opponent cards that went face down, as fillable slots.
   *
   * Four of them in a normal hand, one in Fantasyland -- their fourteenth card.
   * Same panel either way: it is the same question, "what did they throw away",
   * and the same answer, "only if the hand was shown afterwards".
   */
  discardSlots() {
    if (this.fl.opp) {
      return [{ key: "fl", label: "FL", side: this.flHands.opp }];
    }
    return [1, 2, 3, 4].map((turn) => ({
      key: turn,
      label: `T${turn}`,
      side: this.streets[turn].opp,
    }));
  },

  slotFor(key) {
    return this.discardSlots().find((s) => s.key === key) || null;
  },

  /** Slot the next typed card fills, or null if none left.
   *
   * An explicitly chosen slot wins even when it already holds a card -- having
   * arrowed or clicked onto T3, typing a card there should correct T3, not
   * silently land somewhere else.
   */
  discardTarget() {
    if (this.discardSlot !== null && this.slotFor(this.discardSlot)) return this.discardSlot;
    const open = this.discardSlots().find((s) => !s.side.discard);
    return open ? open.key : null;
  },

  /** Drop a half-typed rank and its prompt.
   *
   * Called from every path that ends a card entry, not just the keyboard one:
   * clicking a place button or the undo button never reaches `onKey`, and a
   * buffer left armed there eats the first key of the NEXT card typed.
   */
  clearKeyBuffer() {
    if (!this.keyBuffer) return;
    this.keyBuffer = "";
    $("#hl-status").textContent = "";
  },

  addCard(card) {
    if (this.inDiscardPhase()) {
      // Say why nothing happened: while transcribing, a card that is already
      // on a board is the signal that this one was misread.
      if (this.used().has(card)) {
        toast(`${card} はすでに入力済みです`, "warn");
        return;
      }
      const key = this.discardTarget();
      if (key === null) {
        toast("相手の捨て札はすべて入力済みです（訂正する枠をクリックしてください）", "warn");
        return;
      }
      const side = this.slotFor(key).side;
      side.discard = card;
      side.dealt = [...side.placements.map(([c]) => c), card];
      // Remember the order they were typed in, so backspace undoes the last
      // thing the user did rather than the highest-numbered street.
      this.discardOrder = this.discardOrder.filter((t) => t !== key).concat(key);
      this.discardSlot = null;
      this.render();
      return;
    }

    const info = this.stepInfo();
    if (!info) return;
    if (this.used().has(card)) {
      const side = this.side(info);
      if (side.dealt.includes(card)) this.selected = card;
      this.render();
      return;
    }
    if (this.phase(info) !== "deal") {
      toast("先に入力したカードの置き場所を決めてください", "warn");
      return;
    }
    this.side(info).dealt.push(card);
    this.selected = null;
    this.render();
  },

  place(dest) {
    const info = this.stepInfo();
    if (!info) return;
    if (this.phase(info) === "deal") {
      toast(`まずカードを ${this.dealCount(info)} 枚入れてください`, "warn");
      return;
    }
    const side = this.side(info);
    const pending = this.pending(info);
    const card = this.selected && pending.includes(this.selected) ? this.selected : pending[0];
    if (!card) return;

    if (dest === "discard") {
      if (info.kind === "street" && info.turn === 0) {
        return toast("T0 に捨て札はありません", "warn");
      }
      if (info.seat === "opp") {
        return toast(
          info.kind === "fl"
            ? "相手の14枚目は最後にまとめて入力します"
            : "相手の捨て札は最後にまとめて入力します",
          "warn"
        );
      }
      if (side.discard) return toast("捨て札は1枚だけです", "warn");
      side.discard = card;
    } else {
      if (side.placements.length >= this.placeCount(info)) {
        return toast("この手番の配置は埋まっています", "warn");
      }
      const board = this.board(info.seat);
      if (board[dest].length >= ROW_CAP[dest]) {
        return toast(`${ROW_JP[dest]}段は満杯です`, "warn");
      }
      side.placements.push([card, dest]);
    }
    this.selected = null;
    this.clearKeyBuffer();

    // The hero's own deal leaves exactly one card over -- the third on a
    // street, the fourteenth in Fantasyland -- and naming it explicitly is
    // busywork.  The opponent types only what they placed, so there is
    // nothing left over to name.
    const left = this.pending(info);
    if (
      info.seat === "hero" &&
      (info.kind === "fl" || info.turn > 0) &&
      side.placements.length === this.placeCount(info) &&
      left.length === 1 &&
      !side.discard
    ) {
      side.discard = left[0];
    }
    if (this.pending(info).length === 0) this.step += 1;
    this.render();
  },

  undo() {
    this.clearKeyBuffer();
    // The discard panel is its own undo stack: clear the last opponent discard
    // typed before falling back into the placement walk behind it.
    if (this.inDiscardPhase()) {
      const slots = this.discardSlots();
      const order = this.discardOrder.filter((k) => (this.slotFor(k) || {}).side?.discard);
      const filled = order.length
        ? order[order.length - 1]
        : [...slots].reverse().find((s) => s.side.discard)?.key;
      if (filled !== undefined && filled !== null) {
        const side = this.slotFor(filled).side;
        side.discard = null;
        side.dealt = side.placements.map(([c]) => c);
        this.discardOrder = this.discardOrder.filter((k) => k !== filled);
        this.discardSlot = null;
        this.render();
        return;
      }
      this.step = this.stepList().length - 1;
    }
    // Walk back to the last thing entered, whichever step it was on.
    for (let i = Math.min(this.step, this.stepList().length - 1); i >= 0; i--) {
      const info = this.stepInfo(i);
      const side = this.side(info);
      if (side.discard) { side.discard = null; this.step = i; this.render(); return; }
      if (side.placements.length) { side.placements.pop(); this.step = i; this.render(); return; }
      if (side.dealt.length) { side.dealt.pop(); this.step = i; this.render(); return; }
    }
    toast("戻せる操作がありません");
  },

  reset(keepPosition = true) {
    this.streets = hlEmptyStreets();
    this.flHands = { hero: hlEmptySide(), opp: hlEmptySide() };
    this.step = 0;
    this.selected = null;
    this.discardSlot = null;
    this.discardOrder = [];
    this.editingId = null;
    this.clearKeyBuffer();
    if (!keepPosition) this.heroPosition = "first";
    $("#hl-label").value = "";
    this.render();
  },

  payload() {
    return {
      label: $("#hl-label").value.trim(),
      note: "",
      hero_position: this.heroPosition,
      fl: { hero: this.fl.hero, opp: this.fl.opp },
      fl_hands: Object.fromEntries(
        ["hero", "opp"]
          .filter((seat) => this.fl[seat])
          .map((seat) => [
            seat,
            seat === "hero"
              ? {
                  dealt: this.flHands.hero.dealt,
                  placements: this.flHands.hero.placements,
                  discard: this.flHands.hero.discard,
                }
              : {
                  placements: this.flHands.opp.placements,
                  discard: this.flHands.opp.discard,
                },
          ])
      ),
      // A seat in Fantasyland sends nothing per street; the server rejects a
      // record that tries to have it both ways.
      // The opponent's side goes out as placements + optional discard and the
      // server derives `dealt`, so there is exactly one place that decides
      // what the opponent was holding.
      streets: (this.fl.hero && this.fl.opp ? [] : this.streets).map((street) => ({
        turn: street.turn,
        hero: this.fl.hero
          ? { dealt: [], placements: [], discard: null }
          : {
              dealt: street.hero.dealt,
              placements: street.hero.placements,
              discard: street.hero.discard,
            },
        opp: this.fl.opp
          ? { dealt: [], placements: [], discard: null }
          : {
              placements: street.opp.placements,
              discard: street.opp.discard,
            },
      })),
    };
  },

  complete() {
    return this.step >= this.stepList().length;
  },

  oppDiscardsEntered() {
    return this.discardSlots().filter((s) => s.side.discard).length;
  },

  /** Opponent streets the analysis will skip, mirroring handlog._gradable.
   *
   * A missing discard costs its own street and every later one: the engine
   * checks the running discard count, so T3 is ungradable when T1's card is
   * unknown even if T3's own is not.
   */
  oppUngradableTurns() {
    if (this.fl.opp) return [];
    const out = [];
    let known = 0;
    for (const turn of [1, 2, 3, 4]) {
      if (!this.streets[turn].opp.discard || known !== turn - 1) out.push(turn);
      if (this.streets[turn].opp.discard) known += 1;
    }
    return out;
  },

  // ---------------- rendering ----------------

  render() {
    const info = this.stepInfo();
    $("#hl-progress").textContent = info
      ? `${info.kind === "fl" ? "FL" : "T" + info.turn} / ${
          info.seat === "hero" ? "自分" : "相手"
        }（${info.index + 1}・${this.stepList().length}手番目）`
      : `配置入力ずみ・相手の捨て札 ${this.oppDiscardsEntered()}/4`;

    this.renderStep(info);
    this.renderBoards(info);
    this.renderDiscardPanel();
    this.renderPicker();

    const posLabel = this.heroPosition === "first" ? ["先行", "後攻"] : ["後攻", "先行"];
    $("#hl-hero-pos").textContent = posLabel[0] + (this.fl.hero ? "・FL" : "");
    $("#hl-opp-pos").textContent = posLabel[1] + (this.fl.opp ? "・FL" : "");
    $("#hl-position").value = this.heroPosition;
    $("#hl-pattern").value = this.patternKey();
    // With a seat in Fantasyland the order within a street stops meaning
    // anything for it, but it still selects the evaluator for the other seat.
    $("#hl-position").disabled = this.fl.hero && this.fl.opp;

    $("#hl-save").disabled = !this.complete();
    $("#hl-analyze").disabled = !this.complete();
    $$("#view-handlog .hl-place-btns button[data-place]").forEach((b) => {
      const usable = info && this.phase(info) === "place";
      // The discard key has no meaning on T0, nor on any opponent street: the
      // hero's leftover is derived and the opponent's is entered at the end.
      const discardable = usable && info.seat === "hero" && info.turn > 0;
      b.disabled = b.dataset.place === "discard" ? !discardable : !usable;
    });

    // Grading the opponent's T1-T4 needs the card they threw away, but their
    // T0 is five placements and nothing hidden, so the option stays available
    // whatever the muck situation -- it just says what it will skip.
    const skipped = this.oppUngradableTurns();
    $("#hl-include-opp-note").textContent = skipped.length
      ? `（相手の T${skipped.join(", T")} は捨て札が未入力のため採点をスキップします）`
      : "";
  },

  renderStep(info) {
    const el = $("#hl-step");
    el.innerHTML = "";
    if (!info) {
      el.className = "hl-step complete";
      const edit = this.editingId ? `（#${this.editingId} を編集中）` : "";
      const hint = this.fl.opp
        ? "相手がFLで伏せた14枚目が分かる場合は下で入力すると、相手のセットも採点できます。"
        : "相手の捨て札が分かる場合は下で入力すると、相手の判断も採点できます。";
      el.textContent = `配置はすべて入力済み${edit} — 保存または解析できます。` + hint;
      return;
    }
    el.className = "hl-step " + (info.seat === "hero" ? "hero" : "opp");
    const who = info.seat === "hero" ? "あなた" : "相手";
    const where = info.kind === "fl" ? "FL" : `T${info.turn}`;
    const phase = this.phase(info);
    const side = this.side(info);
    if (phase === "deal") {
      const left = this.dealCount(info) - side.dealt.length;
      const what =
        info.kind === "fl"
          ? info.seat === "hero"
            ? "配られた14枚"
            : "セットした13枚"
          : info.seat === "hero"
          ? "配札"
          : "置いたカード";
      el.textContent = `${where}・${who}の${what}を入力（あと ${left} 枚）— カードをクリック、またはキーボードで「as」「th」`;
    } else {
      const placedLeft = this.placeCount(info) - side.placements.length;
      const tail =
        info.seat === "hero" &&
        (info.kind === "fl" || info.turn > 0) &&
        placedLeft === 1 &&
        !side.discard
          ? "（残り1枚は自動で捨て札）"
          : "";
      el.textContent = `${where}・${who}の配置（あと ${placedLeft} 枚）${tail} — 上(T) / 中(M) / 下(B)`;
    }
  },

  /** The eleventh step: the opponent's four discards, all of them optional. */
  renderDiscardPanel() {
    const panel = $("#hl-discards");
    panel.innerHTML = "";
    if (!this.inDiscardPhase()) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";

    const head = document.createElement("div");
    head.className = "hl-discard-head";
    head.innerHTML =
      (this.fl.opp
        ? "相手がFLで伏せた14枚目（分かる場合のみ）"
        : "相手の捨て札（分かる場合のみ・T1〜T4）") +
      '<span class="muted">　上から順に埋まります。' +
      "枠をクリック、または <kbd>↑</kbd><kbd>↓</kbd> で入力先を移動</span>";
    panel.appendChild(head);

    const target = this.discardTarget();
    for (const slot of this.discardSlots()) {
      const side = slot.side;
      const row = document.createElement("div");
      row.className = "hl-discard-row" + (slot.key === target ? " active" : "");

      const label = document.createElement("span");
      label.className = "hl-discard-label";
      label.textContent = slot.label;
      row.appendChild(label);

      const placed = document.createElement("span");
      placed.className = "hl-discard-placed";
      for (const [card] of side.placements) {
        placed.appendChild(cardEl(card, { small: true, classes: ["hl-discarded"] }));
      }
      row.appendChild(placed);

      const arrow = document.createElement("span");
      arrow.className = "muted";
      arrow.textContent = "＋捨て";
      row.appendChild(arrow);

      if (side.discard) {
        // Clicking a filled slot clears it and aims the next card at it, so a
        // misread card is corrected in two clicks rather than by starting over.
        row.appendChild(
          cardEl(side.discard, {
            small: true,
            onClick: () => {
              side.discard = null;
              side.dealt = side.placements.map(([c]) => c);
              this.discardSlot = slot.key;
              this.render();
            },
          })
        );
      } else {
        const empty = slotEl(true);
        empty.classList.add("clickable");
        empty.addEventListener("click", () => {
          this.discardSlot = slot.key;
          this.render();
        });
        row.appendChild(empty);
      }
      panel.appendChild(row);
    }

    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = this.fl.opp
      ? "相手がFLの局面は自分の街を採点できません（エンジンの信念サンプラーが通常の相手しか想定していないため）。"
      : "未入力でも自分の判断はすべて採点できます（相手の捨て札はレギュラーでは非公開のため、自分の情報集合には含まれません）。";
    panel.appendChild(note);
  },

  renderBoards(info) {
    for (const seat of ["hero", "opp"]) {
      const container = $(seat === "hero" ? "#hl-hero-board" : "#hl-opp-board");
      const board = this.board(seat);
      container.innerHTML = "";
      for (const row of ROWS) {
        const rowEl = document.createElement("div");
        rowEl.className = "row";
        for (const card of board[row]) rowEl.appendChild(cardEl(card, { small: true }));
        for (let i = board[row].length; i < ROW_CAP[row]; i++) rowEl.appendChild(slotEl(true));
        container.appendChild(rowEl);
      }

      const tray = $(seat === "hero" ? "#hl-hero-tray" : "#hl-opp-tray");
      tray.innerHTML = `<span class="tray-label">${
        seat === "hero" ? "自分の配札" : "相手が置いたカード"
      }</span>`;
      const active = info && info.seat === seat;
      const pending = active ? this.pending(info) : [];
      for (const card of pending) {
        const classes = card === this.selected ? ["staged"] : [];
        tray.appendChild(
          cardEl(card, {
            small: true,
            classes,
            onClick: () => { this.selected = card; this.render(); },
          })
        );
      }
      const discards = this.discards(seat);
      if (discards.length) {
        const lab = document.createElement("span");
        lab.className = "tray-label";
        lab.textContent = "捨て";
        tray.appendChild(lab);
        for (const card of discards) {
          tray.appendChild(cardEl(card, { small: true, classes: ["hl-discarded"] }));
        }
      }
    }
  },

  renderPicker() {
    const picker = $("#hl-picker");
    picker.innerHTML = "";
    const used = this.used();
    for (const rank of HL_RANKS) {
      for (const suit of HL_SUITS) {
        const card = rank + suit;
        const classes = used.has(card) ? ["used"] : [];
        picker.appendChild(
          cardEl(card, { classes, onClick: () => this.addCard(card) })
        );
      }
    }
  },

  // ---------------- keyboard ----------------

  onKey(ev) {
    if (!$("#view-handlog").classList.contains("active")) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;

    const key = ev.key.toLowerCase();
    if (key === "backspace") { ev.preventDefault(); this.clearKeyBuffer(); this.undo(); return; }

    const info = this.stepInfo();
    if (info && this.phase(info) === "place") {
      const dest = { t: "top", m: "middle", b: "bottom", x: "discard" }[key];
      if (dest) { ev.preventDefault(); this.clearKeyBuffer(); this.place(dest); return; }
    }
    // In the discard panel the arrows move which street the next card fills,
    // so a half-known muck ("I only saw the T3 one") need not be entered in
    // order.  Digits cannot do this job: 2, 3 and 4 are ranks, and 1 is a ten.
    if (this.inDiscardPhase() && ["arrowdown", "arrowup", "tab"].includes(key)) {
      ev.preventDefault();
      this.clearKeyBuffer();
      const keys = this.discardSlots().map((slot) => slot.key);
      if (keys.length) {
        const current = this.discardSlot !== null ? this.discardSlot : this.discardTarget();
        const at = Math.max(0, keys.indexOf(current));
        const step = key === "arrowup" || (key === "tab" && ev.shiftKey) ? -1 : 1;
        this.discardSlot = keys[(at + step + keys.length) % keys.length];
        this.render();
      }
      return;
    }

    // Card entry: a rank key arms the buffer, the next suit key commits it.
    const rank = key === "1" || key === "0" ? "T" : key.toUpperCase();
    if (HL_RANKS.includes(rank) && !this.keyBuffer) {
      this.keyBuffer = rank;
      $("#hl-status").textContent = `${rankDisp(rank)}… スート（s/h/d/c）を押してください`;
      return;
    }
    if (this.keyBuffer && HL_SUITS.includes(key)) {
      const card = this.keyBuffer + key;
      this.keyBuffer = "";
      $("#hl-status").textContent = "";
      this.addCard(card);
      return;
    }
    if (this.keyBuffer) { this.keyBuffer = ""; $("#hl-status").textContent = ""; }
  },

  // ---------------- persistence ----------------

  async save(silent = false) {
    if (!this.complete()) return null;
    try {
      const path = this.editingId ? `/api/handlog/${this.editingId}` : "/api/handlog";
      const res = await api(path, { method: this.editingId ? "PUT" : "POST", body: this.payload() });
      this.editingId = res.id;
      if (!silent) toast(`保存しました（#${res.id}）`, "good");
      await this.refresh();
      return res.id;
    } catch (e) {
      toast("保存に失敗: " + e.message, "bad");
      return null;
    }
  },

  async refresh() {
    try {
      const res = await api("/api/handlog");
      this.list = res.hands || [];
      this.renderList();
    } catch (e) { /* list is non-critical */ }
  },

  renderList() {
    const el = $("#hl-list");
    el.innerHTML = "";
    if (!this.list.length) {
      el.innerHTML = '<p class="muted">まだ保存されたハンドはありません。</p>';
      return;
    }
    for (const item of this.list) {
      const row = document.createElement("div");
      row.className = "hl-item" + (item.id === this.editingId ? " current" : "");

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = this.checked.has(item.id);
      cb.addEventListener("change", () => {
        if (cb.checked) this.checked.add(item.id); else this.checked.delete(item.id);
      });
      row.appendChild(cb);

      const label = document.createElement("span");
      label.className = "hl-item-label";
      const when = new Date(item.created_at * 1000);
      label.textContent = `#${item.id} ${item.label || when.toLocaleString("ja-JP")}`;
      label.title = "クリックで読み込み";
      label.addEventListener("click", () => this.load(item.id));
      row.appendChild(label);

      const meta = document.createElement("span");
      meta.className = "hl-item-meta";
      meta.innerHTML = item.analyzed_at
        ? `<span class="tag">${item.position === "first" ? "先行" : "後攻"}</span>` +
          `結果 <b>${fmtSigned(item.score, 1)}</b> / EVロス <b class="${item.total_ev_loss >= 3 ? "ev-neg" : ""}">${fmt(item.total_ev_loss, 2)}</b>`
        : `<span class="tag">${item.position === "first" ? "先行" : "後攻"}</span><span class="muted">未解析</span>`;
      row.appendChild(meta);

      const del = document.createElement("button");
      del.className = "small";
      del.textContent = "🗑";
      del.addEventListener("click", async () => {
        if (!confirm(`#${item.id} を削除しますか？`)) return;
        await api(`/api/handlog/${item.id}`, { method: "DELETE" });
        if (this.editingId === item.id) this.editingId = null;
        this.checked.delete(item.id);
        this.refresh();
      });
      row.appendChild(del);

      el.appendChild(row);
    }
  },

  async load(id) {
    // A part-typed hand lives only in this page; loading over it is the one
    // click in the view that can destroy work with nothing to undo it.
    if (!this.editingId && this.used().size) {
      if (!confirm("入力中のハンドが破棄されます。読み込みますか？")) return;
    }
    try {
      const item = await api(`/api/handlog/${id}`);
      this.streets = (item.hand.streets || []).length ? item.hand.streets : hlEmptyStreets();
      this.heroPosition = item.hand.hero_position;
      this.fl = { hero: false, opp: false, ...(item.hand.fl || {}) };
      this.flHands = { hero: hlEmptySide(), opp: hlEmptySide() };
      for (const seat of ["hero", "opp"]) {
        const side = (item.hand.fl_hands || {})[seat];
        if (side) this.flHands[seat] = { dealt: side.dealt || [], placements: side.placements, discard: side.discard || null };
      }
      this.step = this.stepList().length;
      this.editingId = id;
      this.selected = null;
      // Clearing the aimed slot matters: without it a slot chosen while the
      // previous hand was open would swallow the first card typed into this one.
      this.discardSlot = null;
      this.discardOrder = [];
      this.clearKeyBuffer();
      $("#hl-label").value = item.label || "";
      this.render();
      this.renderList();
      if (item.analysis) this.renderResults([{ hand_id: id, label: item.label, analysis: item.analysis }]);
      toast(`#${id} を読み込みました`);
    } catch (e) {
      toast("読み込みに失敗: " + e.message, "bad");
    }
  },

  // ---------------- analysis ----------------

  async analyzeCurrent() {
    const id = await this.save(true);
    if (id === null) return;
    this.startJob({ hand_ids: [id] });
  },

  analyzeSelected() {
    if (!this.checked.size) return toast("解析するハンドを選んでください", "warn");
    this.startJob({ hand_ids: Array.from(this.checked) });
  },

  selectedTurns() {
    return Array.from(document.querySelectorAll(".hl-turn:checked")).map((el) => Number(el.value));
  },

  async startJob(body) {
    if (this.poll) { clearInterval(this.poll); this.poll = null; }
    const precision = $("#hl-precision").value;
    const includeOpp = $("#hl-include-opp").checked;
    const grader = $("#hl-method").value;
    const turns = this.selectedTurns();
    try {
      const res = await api("/api/handlog/analyze", {
        method: "POST",
        body: { ...body, precision, include_opp: includeOpp, method: grader, turns },
      });
      // Per hand: one decision a chosen turn, or the whole five (ten with the
      // opponent).  Only the first frame uses this -- the poll replaces it
      // with the count the job made under the same filter.
      const perHand = (turns.length || 5) * (includeOpp ? 2 : 1);
      this.renderProgress({ status: "running", done: 0, total: res.hands * perHand, hands_done: 0, hands_total: res.hands });
      // A 15-hand deep run is minutes long; a sub-second poll would be
      // thousands of requests to watch a bar that moves once a minute.
      this.poll = setInterval(() => this.pollJob(res.job_id), 1500);
    } catch (e) {
      toast("解析の開始に失敗: " + e.message, "bad");
    }
  },

  async pollJob(jobId) {
    let job;
    try {
      job = await api(`/api/handlog/job/${jobId}`);
    } catch (e) {
      clearInterval(this.poll); this.poll = null;
      toast("解析の取得に失敗: " + e.message, "bad");
      return;
    }
    if (job.status === "running") return this.renderProgress(job);
    clearInterval(this.poll); this.poll = null;
    if (job.status === "error") {
      $("#hl-results").innerHTML = `<p class="bad">解析に失敗しました: ${job.error}</p>`;
      return;
    }
    this.renderResults(job.results || []);
    this.refresh();
  },

  renderProgress(job) {
    const pctDone = job.total ? Math.round((100 * job.done) / job.total) : 0;
    const label = job.current && job.current.label ? `（${job.current.label}）` : "";
    $("#hl-results").innerHTML =
      `<div class="hl-progress"><div class="hl-bar"><span style="width:${pctDone}%"></span></div>` +
      `<p class="muted">解析中 ${job.hands_done}/${job.hands_total} ハンド・${job.done}/${job.total} 判断 ${label}</p></div>`;
  },

  renderResults(results) {
    const area = $("#hl-results");
    area.innerHTML = "";
    if (!results.length) {
      area.innerHTML = '<p class="muted">結果がありません。</p>';
      return;
    }

    if (results.length > 1) {
      const totalLoss = results.reduce((a, r) => a + (r.analysis.hero_total_ev_loss || 0), 0);
      const scores = results.map((r) => (r.analysis.result || {}).score).filter((s) => s != null);
      const sum = document.createElement("div");
      sum.className = "hl-summary";
      sum.innerHTML =
        `<b>${results.length} ハンド合計</b> — EVロス <b>${fmt(totalLoss, 2)}</b>` +
        `（1ハンド ${fmt(totalLoss / results.length, 2)}）` +
        (scores.length
          ? ` / 実結果 ${fmtSigned(scores.reduce((a, b) => a + b, 0), 1)}（1ハンド ${fmtSigned(scores.reduce((a, b) => a + b, 0) / scores.length, 2)}）`
          : "");
      area.appendChild(sum);
    }

    for (const result of results) {
      area.appendChild(this.handResultEl(result));
    }
  },

  /** One hand, replayed street by street with both seats side by side.
   *
   * The table this replaces read like a log: one row per decision, cards as
   * text, the two seats interleaved by deal order. A review is read to answer
   * "what happened, and where did it go wrong", so the shape here follows the
   * hand -- a block per street, both seats inside it, cards drawn as cards --
   * and the candidate list stays folded until it is asked for.
   */
  handResultEl(result) {
    const a = result.analysis;
    const wrap = document.createElement("div");
    wrap.className = "hl-result";

    const byStreet = new Map();
    for (const d of a.decisions) {
      const key = d.kind === "fl" ? "FL" : "T" + d.turn;
      if (!byStreet.has(key)) byStreet.set(key, []);
      byStreet.get(key).push(d);
    }

    const heroLoss =
      a.hero_total_ev_loss != null
        ? a.hero_total_ev_loss
        : a.decisions
            .filter((d) => d.seat === "hero" && d.ev_loss != null)
            .reduce((sum, d) => sum + d.ev_loss, 0);
    const oppGraded = a.decisions.filter((d) => d.seat === "opp" && d.ev_loss != null);
    const worst = a.decisions
      .filter((d) => d.seat === "hero" && d.ev_loss)
      .sort((x, y) => y.ev_loss - x.ev_loss)[0];

    wrap.appendChild(this.summaryEl(result, a, heroLoss, oppGraded, worst));

    for (const [street, decisions] of byStreet) {
      const block = document.createElement("div");
      block.className = "hl-street";

      const label = document.createElement("div");
      label.className = "hl-street-label";
      label.textContent = street;
      block.appendChild(label);

      const body = document.createElement("div");
      body.className = "hl-street-body";
      const cands = document.createElement("div");
      cands.className = "hl-street-cands";

      for (const d of decisions) body.appendChild(this.decisionRow(d, cands, worst));
      block.appendChild(body);
      block.appendChild(cands);
      wrap.appendChild(block);
    }

    for (const note of this.resultNotes(a)) wrap.appendChild(note);
    if ((a.result || {}).hero_board) wrap.appendChild(this.finalBoardsEl(a.result));
    return wrap;
  },

  summaryEl(result, a, heroLoss, oppGraded, worst) {
    const res = a.result || {};
    const grader =
      a.method === "model" ? "モデル" : a.method === "teacher" ? "教師探索" : "教師探索（旧記録）";
    const box = document.createElement("div");
    box.className = "hl-summary-card";

    const title = document.createElement("div");
    title.className = "hl-summary-title";
    title.innerHTML =
      "<b>" + (result.label ? result.label : "#" + result.hand_id) + "</b>" +
      '<span class="tag">' + grader + "</span>" +
      (res.score != null
        ? '<span class="hl-score ' + (res.score >= 0 ? "ev-best" : "ev-neg") + '">実結果 ' +
          fmtSigned(res.score, 1) + "</span>"
        : "");
    box.appendChild(title);

    const cell = (label, value, cls) =>
      '<div class="hl-stat"><span class="hl-stat-label">' + label + "</span>" +
      '<span class="hl-stat-value ' + (cls || "") + '">' + value + "</span></div>";

    const stats = document.createElement("div");
    stats.className = "hl-summary-stats";
    stats.innerHTML =
      cell("自分のEVロス", fmt(heroLoss, 2), heroLoss >= 1 ? "ev-neg" : "ev-best") +
      cell("最善だった判断", a.hero_best + "/" + a.hero_graded) +
      (oppGraded.length
        ? cell("相手のEVロス", fmt(oppGraded.reduce((s, d) => s + d.ev_loss, 0), 2))
        : "") +
      (worst
        ? cell(
            "最大の取りこぼし",
            (worst.kind === "fl" ? "FL" : "T" + worst.turn) + " −" + fmt(worst.ev_loss, 2),
            "ev-neg"
          )
        : "") +
      (res.hero_bust ? cell("自分", "バースト", "ev-neg") : "") +
      (res.opp_bust ? cell("相手", "バースト", "") : "") +
      (res.hero_fl ? cell("自分", "FL獲得", "ev-best") : "") +
      (res.opp_fl ? cell("相手", "FL獲得", "ev-neg") : "");
    box.appendChild(stats);
    return box;
  },

  /** One seat's decision on one street: what they held, what they did, how it graded. */
  decisionRow(d, candsHost, worst) {
    const row = document.createElement("div");
    row.className = "hl-decision " + (d.seat === "hero" ? "hero" : "opp");
    if (d === worst) row.classList.add("worst");

    const who = document.createElement("span");
    who.className = "hl-who";
    who.textContent = d.seat === "hero" ? "自分" : "相手";
    row.appendChild(who);

    const dealt = document.createElement("span");
    dealt.className = "hl-cards";
    for (const card of d.dealt || []) dealt.appendChild(cardEl(card, { small: true }));
    row.appendChild(dealt);

    const arrow = document.createElement("span");
    arrow.className = "hl-arrow";
    arrow.textContent = "→";
    row.appendChild(arrow);

    const play = document.createElement("span");
    play.className = "hl-play";
    play.appendChild(actionText(d.action));
    row.appendChild(play);

    const grade = document.createElement("span");
    grade.className = "hl-grade";
    if (d.error) {
      grade.innerHTML = '<span class="muted" title="' + d.error + '">採点できず</span>';
    } else if (d.fouled) {
      grade.innerHTML = '<span class="ev-neg">バースト</span>';
    } else if (d.rank == null) {
      grade.innerHTML = '<span class="muted">候補外</span>';
    } else {
      const badge = d.is_best
        ? '<span class="hl-rank best">1位 / ' + d.candidate_count + "</span>"
        : '<span class="hl-rank ' + (d.ev_loss >= 1 ? "bad" : "") + '">' +
          d.rank + "位 / " + d.candidate_count + "</span>";
      const loss = d.is_best
        ? '<span class="ev-best">best</span>'
        : d.tied_with_best
        ? '<span class="muted">同率</span>'
        : '<span class="ev-neg">−' + fmt(d.ev_loss) + "</span>";
      grade.innerHTML = badge + loss;
    }
    if (d.vs_fl_approx) {
      grade.innerHTML +=
        '<span class="muted" title="相手がFLであることを知らないT0先行モデルによる近似">※</span>';
    }
    // The model charged something it cannot actually resolve, so the search
    // was asked. Shown rather than swallowed: a user who saw the old number
    // should be able to tell that it was checked, not that it drifted.
    if (d.model_ev_loss != null) {
      const claimed = fmt(d.model_ev_loss);
      grade.innerHTML += d.overturned
        ? '<span class="hl-checked ok" title="モデルは −' + claimed +
          " と評価しましたが、探索は差なしと判定しました（" +
          d.confirm_precision + '）">探索で確認: 差なし</span>'
        : '<span class="hl-checked" title="モデルは −' + claimed +
          " と評価。探索（" + d.confirm_precision + "）で再採点しました\">探索で確認</span>";
    }
    row.appendChild(grade);

    if ((d.candidates || []).length) {
      const button = document.createElement("button");
      button.className = "small hl-expand";
      button.textContent = "候補";
      const id = d.seat + ":" + (d.kind === "fl" ? "FL" : d.turn);
      button.addEventListener("click", () => {
        const open = candsHost.dataset.open === id;
        candsHost.innerHTML = "";
        candsHost.dataset.open = open ? "" : id;
        if (!open) this.showCandidates(d, candsHost);
      });
      row.appendChild(button);
    }

    if (!d.is_best && d.best && d.rank != null && !d.fouled) {
      const better = document.createElement("span");
      better.className = "hl-better";
      better.appendChild(document.createTextNode("最善: "));
      better.appendChild(actionText(d.best.action));
      row.appendChild(better);
    }
    return row;
  },

  finalBoardsEl(res) {
    const box = document.createElement("div");
    box.className = "hl-final";
    const head = document.createElement("div");
    head.className = "hl-final-head";
    head.textContent = "最終盤面";
    box.appendChild(head);
    for (const [key, label] of [["hero_board", "自分"], ["opp_board", "相手"]]) {
      const line = document.createElement("div");
      line.className = "hl-final-line";
      const tag = document.createElement("span");
      tag.className = "hl-who";
      tag.textContent = label;
      line.appendChild(tag);
      for (const row of ROWS) {
        const group = document.createElement("span");
        group.className = "hl-cards";
        for (const card of (res[key] || {})[row] || []) {
          group.appendChild(cardEl(card, { small: true }));
        }
        line.appendChild(group);
      }
      box.appendChild(line);
    }
    return box;
  },

  /** The warnings that qualify a whole hand's grades. */
  resultNotes(a) {
    const notes = [];
    const add = (html) => {
      const p = document.createElement("p");
      p.className = "muted";
      p.innerHTML = html;
      notes.push(p);
    };
    if ((a.fallback_decisions || []).length) {
      const where = a.fallback_decisions
        .map((d) => (d.seat === "hero" ? "自分" : "相手") + "T" + d.turn)
        .join(", ");
      add(
        '<b class="ev-neg">' + where + "</b> はエンジンが拒否したためMCフォールバックで採点されています" +
          "（他の行とスコアの尺度が違います）。"
      );
    }
    if (a.vs_fl_ungraded) {
      const who = (a.fl || {}).opp ? "相手がFLのため、自分" : "自分がFLのため、相手";
      const hasApprox = (a.vs_fl_approx_decisions || []).length > 0;
      add(
        who + "の" + (hasApprox ? "T1以降の" : "") + "街の判断は採点していません" +
          "（m3エンジンの信念サンプラーが通常の相手の盤面しか受け付けないため）。" +
          (hasApprox ? "※印のT0はFLを知らないT0先行モデルによる近似採点です。" : "")
      );
    }
    if ((a.opp_skipped_turns || []).length) {
      add("相手の T" + a.opp_skipped_turns.join(", T") + " は捨て札が未入力のため採点していません。");
    }
    if (a.hero_unmatched) {
      add(a.hero_unmatched + " 件が候補リストに見つかりませんでした（実質同一盤面の可能性があります）。");
    }
    return notes;
  },

  showCandidates(d, detail) {
    if (!detail) return;
    detail.innerHTML = "";
    if (d.error) {
      detail.innerHTML += `<p class="bad">${d.error}</p>`;
      return;
    }
    const table = document.createElement("table");
    table.className = "cands";
    const rows = d.candidates || [];
    // The engine emits bust/FL rates only from the Monte-Carlo path; under the
    // model and the teacher every cell is "-". Two columns of dashes are not
    // information, so they appear only when something fills them.
    const hasRates = rows.some(
      (c) => (c.metrics || {}).bust_rate != null || (c.metrics || {}).fl_rate != null
    );
    const rateCells = (metrics) =>
      hasRates
        ? `<td>${pct((metrics || {}).bust_rate)}</td><td>${pct((metrics || {}).fl_rate)}</td>`
        : "";
    table.innerHTML =
      "<thead><tr><th>#</th><th style='text-align:left'>アクション</th><th>EV差</th>" +
      (hasRates ? "<th>バースト</th><th>FL</th>" : "") +
      "</tr></thead>";
    const tbody = document.createElement("tbody");

    const head = document.createElement("h4");
    const where = d.kind === "fl" ? "FL" : `T${d.turn}`;
    head.textContent =
      `${where}・${d.seat === "hero" ? "自分" : "相手"}` +
      `（${d.position === "first" ? "先行" : "後攻"}）配札 ${d.dealt.join(" ")}` +
      ` — 候補 ${d.candidate_count}（上位 ${rows.length} を表示）`;
    detail.appendChild(head);
    const bestEv = rows[0] && rankScore(rows[0]);
    rows.forEach((cand, i) => {
      const tr = document.createElement("tr");
      if (i === 0) tr.classList.add("best-row");
      if (d.rank === i + 1) tr.classList.add("user-row");
      tr.innerHTML = `<td>${i + 1}</td>`;
      const tdA = document.createElement("td");
      tdA.appendChild(actionText(cand.action));
      tr.appendChild(tdA);
      const delta = rankScore(cand) - bestEv;
      tr.innerHTML +=
        (i === 0 || delta > -0.005 ? `<td class="ev-best">best</td>` : `<td class="ev-neg">${fmt(delta)}</td>`) +
        rateCells(cand.metrics);
      tbody.appendChild(tr);
    });
    // T0 has 232 candidates but only the top 20 are stored; a move ranked
    // below that would otherwise vanish from its own detail view.
    if (d.played && d.rank > rows.length) {
      const tr = document.createElement("tr");
      tr.classList.add("user-row");
      tr.innerHTML = `<td>${d.rank}</td>`;
      const tdA = document.createElement("td");
      tdA.appendChild(actionText(d.played.action));
      tr.appendChild(tdA);
      tr.innerHTML += `<td class="ev-neg">${fmt(-d.ev_loss)}</td>` + rateCells(d.played.metrics);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    detail.appendChild(table);
    detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  },
};

// ---------------- wiring ----------------

$("#hl-position").addEventListener("change", (ev) => {
  const used = HandLog.used().size;
  if (used && !confirm("手番を変えると入力済みの内容が消えます。よろしいですか？")) {
    ev.target.value = HandLog.heroPosition;
    return;
  }
  HandLog.heroPosition = ev.target.value;
  HandLog.reset(true);
});

$("#hl-pattern").addEventListener("change", (ev) => {
  const used = HandLog.used().size;
  if (used && !confirm("種類を変えると入力済みの内容が消えます。よろしいですか？")) {
    ev.target.value = HandLog.patternKey();
    return;
  }
  HandLog.setPattern(ev.target.value);
});

$("#hl-reset").addEventListener("click", () => {
  if (HandLog.used().size && !confirm("入力内容を消して最初からやり直しますか？")) return;
  HandLog.reset(true);
});

$$("#view-handlog .hl-place-btns button[data-place]").forEach((btn) => {
  btn.addEventListener("click", () => HandLog.place(btn.dataset.place));
});

$("#hl-undo").addEventListener("click", () => HandLog.undo());
$("#hl-save").addEventListener("click", () => HandLog.save());
$("#hl-analyze").addEventListener("click", () => HandLog.analyzeCurrent());
$("#hl-analyze-selected").addEventListener("click", () => HandLog.analyzeSelected());
// Precision is a particle budget. The model does not sample, so the control is
// disabled rather than left looking as if it were doing something.
function hlSyncMethod() {
  const isModel = $("#hl-method").value === "model";
  $("#hl-precision").disabled = isModel;
  $("#hl-method-note").textContent = isModel
    ? "決定的：同じ局面なら毎回同じ順位とEV差。1ハンド 1〜2秒。"
    : "シード依存：同じ局面でも順位が動きます（T0で5シード＝5通りの1位）。";
}
$("#hl-method").addEventListener("change", hlSyncMethod);
hlSyncMethod();

// Naming turns is what makes the expensive rungs usable: `deep_t0` is about
// six minutes over a whole hand and most of that is the opening, so a spot
// worth a hard look should not drag the other nine decisions behind it.
function hlSyncTurns() {
  const picked = HandLog.selectedTurns();
  $("#hl-turns-note").textContent = picked.length
    ? `T${picked.join("・T")} だけを採点します`
    : "未選択なら全ターン";
}
$$("#view-handlog .hl-turn").forEach((el) => el.addEventListener("change", hlSyncTurns));
hlSyncTurns();

$("#hl-refresh").addEventListener("click", () => HandLog.refresh());
$("#hl-select-all").addEventListener("click", () => {
  HandLog.list.forEach((h) => HandLog.checked.add(h.id));
  HandLog.renderList();
});
$("#hl-select-none").addEventListener("click", () => {
  HandLog.checked.clear();
  HandLog.renderList();
});

document.addEventListener("keydown", (ev) => HandLog.onKey(ev));

HandLog.render();
HandLog.refresh();
