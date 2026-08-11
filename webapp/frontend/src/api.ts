import {
  EMPTY_BOARD,
  type Board,
  type CardCode,
  type CreateMatchPayload,
  type HandResult,
  type HandSummary,
  type HandView,
  type AiDecisionMeta,
  type MatchView,
  type MetaView,
  type PlacementPayload,
  type Positions,
  type ReplayStep,
  type RowName,
  type Street
} from "./types";

const TOKEN_KEY = "ofc.sharedToken";
const RECENT_KEY = "ofc.recentMatches";
const API_ROOT = String(import.meta.env.VITE_API_ROOT ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY)?.trim() ?? "";
}

export function setToken(token: string): void {
  const clean = token.trim();
  if (clean) localStorage.setItem(TOKEN_KEY, clean);
  else localStorage.removeItem(TOKEN_KEY);
}

export function getRecentMatchIds(): string[] {
  try {
    const value = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string").slice(0, 8) : [];
  } catch {
    return [];
  }
}

export function rememberMatch(id: string): void {
  const next = [id, ...getRecentMatchIds().filter((saved) => saved !== id)].slice(0, 8);
  localStorage.setItem(RECENT_KEY, JSON.stringify(next));
}

export function forgetMatch(id: string): void {
  localStorage.setItem(RECENT_KEY, JSON.stringify(getRecentMatchIds().filter((saved) => saved !== id)));
}

function emptyBoard(): Board {
  return { top: [], middle: [], bottom: [] };
}

function cloneBoard(board: Board): Board {
  return {
    top: [...board.top],
    middle: [...board.middle],
    bottom: [...board.bottom]
  };
}

function cards(value: unknown): CardCode[] {
  return Array.isArray(value)
    ? value
        .map((card) => (typeof card === "string" ? card : String((card as { code?: unknown })?.code ?? "")))
        .filter(Boolean)
    : [];
}

function normalizeBoard(value: unknown): Board {
  const raw = (value ?? {}) as Record<string, unknown>;
  return {
    top: cards(raw.top ?? raw.front),
    middle: cards(raw.middle ?? raw.mid),
    bottom: cards(raw.bottom ?? raw.back)
  };
}

function normalizePositions(value: unknown): Positions {
  const raw = (value ?? {}) as Record<string, unknown>;
  const human = raw.human === "second" ? "second" : "first";
  const ai = raw.ai === "first" ? "first" : "second";
  return { human, ai };
}

function numberValue(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalNumber(value: unknown): number | undefined {
  if (value == null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function summaryRoyaltyTotal(value: unknown): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : 0;
  if (!value || typeof value !== "object") return 0;
  const raw = value as Record<string, unknown>;
  const total = optionalNumber(raw.total);
  if (total != null) return total;
  return (["top", "middle", "bottom"] as const).reduce(
    (sum, row) => sum + (optionalNumber(raw[row]) ?? 0),
    0
  );
}

function normalizeResult(value: unknown): HandResult | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const stacks = (raw.stacks_after ?? {}) as Record<string, unknown>;
  const rawComponents = raw.components ?? raw.point_components;
  return {
    row_wins: raw.row_wins as HandResult["row_wins"],
    scoop: (raw.scoop as HandResult["scoop"]) ?? null,
    royalties: raw.royalties as HandResult["royalties"],
    fouls: raw.fouls as HandResult["fouls"],
    fl_entries: raw.fl_entries as HandResult["fl_entries"],
    components:
      rawComponents && typeof rawComponents === "object"
        ? {
            perspective:
              (rawComponents as Record<string, unknown>).perspective === "human"
                ? "human"
                : undefined,
            foul_base: optionalNumber((rawComponents as Record<string, unknown>).foul_base),
            line_total: optionalNumber((rawComponents as Record<string, unknown>).line_total),
            scoop_bonus: optionalNumber((rawComponents as Record<string, unknown>).scoop_bonus),
            royalty_delta: optionalNumber((rawComponents as Record<string, unknown>).royalty_delta),
            total: optionalNumber((rawComponents as Record<string, unknown>).total)
          }
        : undefined,
    raw_score: numberValue(raw.human_raw_score ?? raw.raw_score ?? raw.hu_score),
    capped_score: numberValue(
      raw.human_capped_score ?? raw.capped_score ?? raw.human_raw_score ?? raw.raw_score ?? raw.hu_score
    ),
    stacks_after: {
      human: numberValue(stacks.human, 200),
      ai: numberValue(stacks.ai, 200)
    }
  };
}

function normalizeSummary(value: unknown, fallbackIndex: number): HandSummary {
  const raw = (value ?? {}) as Record<string, unknown>;
  const result = normalizeResult(raw.result ?? raw.result_json);
  return {
    id: String(raw.id ?? raw.hand_id ?? `hand-${fallbackIndex}`),
    index: numberValue(raw.index, fallbackIndex),
    status: raw.status === "playing" ? "playing" : "complete",
    positions: raw.positions ? normalizePositions(raw.positions) : undefined,
    score: raw.score == null ? result?.raw_score : numberValue(raw.score),
    capped_score: raw.capped_score == null ? result?.capped_score : numberValue(raw.capped_score),
    stacks_after: result?.stacks_after,
    royalties: result
      ? {
          human: summaryRoyaltyTotal(result.royalties?.human),
          ai: summaryRoyaltyTotal(result.royalties?.ai)
        }
      : undefined,
    fl_human: (raw.fl_human ?? (raw.fl_status as Record<string, unknown> | undefined)?.human) as boolean | number,
    fl_ai: (raw.fl_ai ?? (raw.fl_status as Record<string, unknown> | undefined)?.ai) as boolean | number
  };
}

function normalizeReplay(value: unknown, finalBoards: { human: Board; ai: Board }): ReplayStep[] {
  if (!Array.isArray(value)) return [];
  return value.map((entry, index) => {
    const raw = (entry ?? {}) as Record<string, unknown>;
    const rawBoards = (raw.boards ?? {}) as Record<string, unknown>;
    const actor = raw.actor === "ai" || raw.actor === "human" ? raw.actor : "system";
    const street = typeof raw.street === "string" ? (raw.street as Street) : null;
    return {
      id: String(raw.id ?? `step-${index}`),
      index: numberValue(raw.index, index),
      actor,
      street,
      label: String(raw.label ?? `${street ?? "結果"} · ${actor === "human" ? "あなた" : actor === "ai" ? "AI" : "システム"}`),
      boards: {
        human: rawBoards.human ? normalizeBoard(rawBoards.human) : cloneBoard(finalBoards.human),
        ai: rawBoards.ai ? normalizeBoard(rawBoards.ai) : cloneBoard(finalBoards.ai)
      },
      dealt_cards: cards(raw.dealt_cards ?? raw.dealt),
      discards: cards(raw.discards),
      think_ms: raw.think_ms == null ? undefined : numberValue(raw.think_ms),
      ai_meta: normalizeAiMeta(raw.ai_meta ?? raw.ai_meta_json)
    };
  });
}

function normalizeAiMeta(value: unknown): AiDecisionMeta | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const topk = Array.isArray(raw.scores_topk)
    ? raw.scores_topk.map((candidate, index) => {
        const item = (candidate ?? {}) as Record<string, unknown>;
        const rawScore = item.score ?? item.value;
        const score = rawScore == null || !Number.isFinite(Number(rawScore)) ? null : Number(rawScore);
        return {
          rank: item.rank == null ? index + 1 : numberValue(item.rank, index + 1),
          score,
          placements: Array.isArray(item.placements)
            ? (item.placements as Array<[CardCode, RowName]>)
            : undefined,
          discards: cards(item.discards)
        };
      })
    : undefined;
  const weightsRaw = raw.weights_sha;
  const weightsSha = Array.isArray(weightsRaw)
    ? weightsRaw.map(String)
    : weightsRaw == null
      ? undefined
      : String(weightsRaw);
  return {
    evaluator: raw.evaluator == null ? undefined : String(raw.evaluator),
    weights_sha: weightsSha,
    assembly_sha: raw.assembly_sha == null ? undefined : String(raw.assembly_sha),
    scores_topk: topk
  };
}

export function normalizeMatch(value: unknown): MatchView {
  const raw = (value ?? {}) as Record<string, unknown>;
  const stacks = (raw.stacks ?? {}) as Record<string, unknown>;
  const handsRaw = Array.isArray(raw.hands) ? raw.hands : [];
  const statusValue = String(raw.status ?? "ready");
  const status: MatchView["status"] =
    statusValue === "in_hand" || statusValue === "awaiting_continue" || statusValue === "completed"
      ? statusValue
      : "ready";
  return {
    id: String(raw.id ?? raw.match_id ?? ""),
    status,
    stacks: {
      human: numberValue(stacks.human ?? raw.human_stack, 200),
      ai: numberValue(stacks.ai ?? raw.ai_stack, 200)
    },
    first_hand_positions: normalizePositions(raw.first_hand_positions ?? raw.first_positions),
    current_hand_id: raw.current_hand_id == null ? null : String(raw.current_hand_id),
    can_start_hand: Boolean(raw.can_start_hand ?? status === "ready"),
    can_continue: Boolean(raw.can_continue ?? status === "awaiting_continue"),
    hands: handsRaw.map(normalizeSummary),
    assembly_sha: raw.assembly_sha == null ? undefined : String(raw.assembly_sha),
    created_at: raw.created_at == null ? undefined : String(raw.created_at)
  };
}

export function normalizeHand(value: unknown): HandView {
  const raw = (value ?? {}) as Record<string, unknown>;
  const rawBoards = (raw.boards ?? {}) as Record<string, unknown>;
  const boards = {
    human: normalizeBoard(rawBoards.human ?? raw.human_board),
    ai: normalizeBoard(rawBoards.ai ?? raw.ai_board ?? raw.opponent_board)
  };
  const flRaw = (raw.fl_status ?? {}) as Record<string, unknown>;
  const actionValue = raw.action_required;
  const actionRequired: HandView["action_required"] =
    actionValue === "fl" || actionValue === "normal" ? actionValue : null;
  const streetValue = raw.street;
  const street = typeof streetValue === "string" ? (streetValue as Street) : null;
  const toAct = raw.to_act === "human" || raw.to_act === "ai" ? raw.to_act : null;
  return {
    id: String(raw.id ?? raw.hand_id ?? ""),
    match_id: String(raw.match_id ?? ""),
    index: numberValue(raw.index, 1),
    status: raw.status === "complete" || raw.result ? "complete" : "playing",
    positions: normalizePositions(raw.positions),
    fl_status: {
      human: (flRaw.human ?? raw.fl_human ?? false) as boolean | number,
      ai: (flRaw.ai ?? raw.fl_ai ?? false) as boolean | number
    },
    street,
    to_act: toAct,
    boards,
    dealt_cards: cards(raw.dealt_cards ?? raw.human_cards ?? raw.cards),
    action_required: actionRequired,
    result: normalizeResult(raw.result ?? raw.result_json),
    ai_pending: Boolean(raw.ai_pending),
    replay: normalizeReplay(raw.replay ?? raw.timeline ?? raw.decisions, boards)
  };
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_ROOT}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = await response.json();
    } catch {
      detail = await response.text();
    }
    const message =
      typeof detail === "object" && detail && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `サーバーエラー (${response.status})`;
    throw new ApiError(message, response.status, detail);
  }
  return (await response.json()) as T;
}

function usingDemo(): boolean {
  return getToken() === "demo" || String(import.meta.env.VITE_DEMO_MODE ?? "") === "true";
}

const demoStreetCards: Record<Street, CardCode[]> = {
  T0: ["AS", "KH", "QD", "JC", "9S"],
  T1: ["AH", "8D", "3C"],
  T2: ["KS", "7H", "4D"],
  T3: ["QS", "10H", "5C"],
  T4: ["JH", "9D", "2S"],
  FL: ["AS", "AH", "AD", "KS", "KH", "KD", "QS", "QH", "QD", "JC", "10C", "9C", "8C", "2D"]
};

let demoFl = false;
let demoMatch: MatchView = makeDemoMatch();
let demoHand: HandView = makeDemoHand(false);

function makeDemoMatch(): MatchView {
  return {
    id: "demo-match",
    status: "ready",
    stacks: { human: 200, ai: 200 },
    first_hand_positions: { human: "first", ai: "second" },
    current_hand_id: null,
    can_start_hand: true,
    can_continue: false,
    hands: [],
    assembly_sha: "demo-stage19"
  };
}

function makeDemoHand(fl: boolean): HandView {
  return {
    id: "demo-hand-1",
    match_id: "demo-match",
    index: 1,
    status: "playing",
    positions: { human: "first", ai: "second" },
    fl_status: { human: fl ? 14 : false, ai: false },
    street: fl ? "FL" : "T0",
    to_act: "human",
    boards: { human: emptyBoard(), ai: emptyBoard() },
    dealt_cards: [...(fl ? demoStreetCards.FL : demoStreetCards.T0)],
    action_required: fl ? "fl" : "normal",
    result: null,
    ai_pending: false,
    replay: []
  };
}

function demoResult(): HandResult {
  return {
    row_wins: {
      top: "not_scored_foul",
      middle: "not_scored_foul",
      bottom: "not_scored_foul"
    },
    scoop: null,
    royalties: { human: 8, ai: 0 },
    fouls: { human: false, ai: true },
    fl_entries: { human: 14, ai: false },
    components: {
      perspective: "human",
      foul_base: 6,
      line_total: 0,
      scoop_bonus: 0,
      royalty_delta: 8,
      total: 14
    },
    raw_score: 14,
    capped_score: 14,
    stacks_after: { human: 214, ai: 186 }
  };
}

function addOpponentCards(street: Street, board: Board): void {
  const byStreet: Partial<Record<Street, Array<[CardCode, RowName]>>> = {
    T0: [
      ["10S", "bottom"],
      ["10D", "bottom"],
      ["8S", "middle"],
      ["7S", "middle"],
      ["4C", "top"]
    ],
    T1: [
      ["6S", "middle"],
      ["5S", "middle"]
    ],
    T2: [
      ["3S", "middle"],
      ["9H", "bottom"]
    ],
    T3: [
      ["9C", "bottom"],
      ["2H", "top"]
    ],
    T4: [
      ["6D", "bottom"],
      ["3H", "top"]
    ]
  };
  for (const [card, row] of byStreet[street] ?? []) board[row].push(card);
}

function recordDemoStep(actor: "human" | "ai", street: Street, label: string): void {
  demoHand.replay.push({
    id: `${demoHand.id}-${demoHand.replay.length + 1}`,
    index: demoHand.replay.length,
    actor,
    street,
    label,
    boards: {
      human: cloneBoard(demoHand.boards.human),
      ai: cloneBoard(demoHand.boards.ai)
    },
    think_ms: actor === "ai" ? 284 : undefined,
    ai_meta:
      actor === "ai"
        ? {
            evaluator: "demo-stage19",
            weights_sha: "demo…8f31",
            scores_topk: [
              { rank: 1, value: 2.48 },
              { rank: 2, value: 2.31 },
              { rank: 3, value: 2.16 }
            ]
          }
        : null
  });
}

async function demoDelay<T>(value: T, ms = 180): Promise<T> {
  await new Promise((resolve) => window.setTimeout(resolve, ms));
  return structuredClone(value);
}

async function demoCreateMatch(payload: CreateMatchPayload): Promise<MatchView> {
  demoFl = payload.seed === 1717;
  demoMatch = makeDemoMatch();
  demoHand = makeDemoHand(demoFl);
  return demoDelay(demoMatch);
}

async function demoStartHand(): Promise<HandView> {
  demoMatch.status = "in_hand";
  demoMatch.current_hand_id = demoHand.id;
  demoMatch.can_start_hand = false;
  return demoDelay(demoHand);
}

async function demoAction(payload: PlacementPayload): Promise<HandView> {
  const street = demoHand.street ?? "T0";
  for (const [card, row] of payload.placements) demoHand.boards.human[row].push(card);
  recordDemoStep("human", street, `${street} · あなたの配置`);
  addOpponentCards(street, demoHand.boards.ai);
  recordDemoStep("ai", street, `${street} · AIの配置`);
  if (street === "T4") {
    demoHand.status = "complete";
    demoHand.to_act = null;
    demoHand.action_required = null;
    demoHand.dealt_cards = [];
    demoHand.result = demoResult();
    demoMatch.status = "awaiting_continue";
    demoMatch.can_continue = true;
    demoMatch.stacks = { ...demoHand.result.stacks_after };
    const summary = normalizeSummary(
      { id: demoHand.id, index: 1, status: "complete", result: demoHand.result },
      1
    );
    demoMatch.hands = [summary];
  } else {
    const next = `T${Number(street.slice(1)) + 1}` as Street;
    demoHand.street = next;
    demoHand.dealt_cards = [...demoStreetCards[next]];
  }
  return demoDelay(demoHand, 320);
}

async function demoFlPlacement(payload: PlacementPayload): Promise<HandView> {
  for (const [card, row] of payload.placements) demoHand.boards.human[row].push(card);
  addOpponentCards("T0", demoHand.boards.ai);
  addOpponentCards("T1", demoHand.boards.ai);
  addOpponentCards("T2", demoHand.boards.ai);
  addOpponentCards("T3", demoHand.boards.ai);
  addOpponentCards("T4", demoHand.boards.ai);
  recordDemoStep("human", "FL", "FL · あなたの一括配置");
  demoHand.status = "complete";
  demoHand.to_act = null;
  demoHand.action_required = null;
  demoHand.dealt_cards = [];
  demoHand.result = demoResult();
  demoMatch.status = "awaiting_continue";
  demoMatch.can_continue = true;
  demoMatch.stacks = { ...demoHand.result.stacks_after };
  demoMatch.hands = [
    normalizeSummary({ id: demoHand.id, index: 1, status: "complete", result: demoHand.result }, 1)
  ];
  return demoDelay(demoHand, 340);
}

export const api = {
  async createMatch(payload: CreateMatchPayload = {}): Promise<MatchView> {
    const match = usingDemo()
      ? await demoCreateMatch(payload)
      : normalizeMatch(await requestJson<unknown>("/api/match", { method: "POST", body: JSON.stringify(payload) }));
    rememberMatch(match.id);
    return match;
  },

  async getMatch(id: string): Promise<MatchView> {
    return usingDemo() && id === demoMatch.id
      ? demoDelay(demoMatch)
      : normalizeMatch(await requestJson<unknown>(`/api/match/${encodeURIComponent(id)}`));
  },

  async startHand(matchId: string): Promise<HandView> {
    return usingDemo() && matchId === demoMatch.id
      ? demoStartHand()
      : normalizeHand(
          await requestJson<unknown>(`/api/match/${encodeURIComponent(matchId)}/hand`, { method: "POST" })
        );
  },

  async getHand(id: string, signal?: AbortSignal): Promise<HandView> {
    return usingDemo() && id === demoHand.id
      ? demoDelay(demoHand, 80)
      : normalizeHand(await requestJson<unknown>(`/api/hand/${encodeURIComponent(id)}`, { signal }));
  },

  async submitAction(id: string, payload: PlacementPayload): Promise<HandView> {
    return usingDemo() && id === demoHand.id
      ? demoAction(payload)
      : normalizeHand(
          await requestJson<unknown>(`/api/hand/${encodeURIComponent(id)}/action`, {
            method: "POST",
            body: JSON.stringify(payload)
          })
        );
  },

  async submitFlPlacement(id: string, payload: PlacementPayload): Promise<HandView> {
    return usingDemo() && id === demoHand.id
      ? demoFlPlacement(payload)
      : normalizeHand(
          await requestJson<unknown>(`/api/hand/${encodeURIComponent(id)}/fl-placement`, {
            method: "POST",
            body: JSON.stringify(payload)
          })
        );
  },

  async continueMatch(id: string, shouldContinue: boolean): Promise<MatchView> {
    if (usingDemo() && id === demoMatch.id) {
      demoMatch.status = shouldContinue ? "ready" : "completed";
      demoMatch.can_continue = false;
      demoMatch.can_start_hand = shouldContinue;
      if (!shouldContinue) demoMatch.current_hand_id = null;
      return demoDelay(demoMatch);
    }
    return normalizeMatch(
      await requestJson<unknown>(`/api/match/${encodeURIComponent(id)}/continue`, {
        method: "POST",
        body: JSON.stringify({ continue: shouldContinue })
      })
    );
  },

  async getMeta(): Promise<MetaView> {
    if (usingDemo()) {
      return demoDelay({
        app_version: "demo",
        assembly_sha: "demo-stage19",
        assembly: { T0: "stage19_p0", T1: "stage19_p0", T4: "exact" }
      });
    }
    return requestJson<MetaView>("/api/meta");
  },

  async exportMatch(id: string): Promise<Blob> {
    if (usingDemo() && id === demoMatch.id) {
      const lines = [
        ...demoHand.replay.map((step) => JSON.stringify({ type: "decision", hand_id: demoHand.id, ...step })),
        JSON.stringify({ type: "hand_summary", hand_id: demoHand.id, result: demoHand.result })
      ];
      return new Blob([`${lines.join("\n")}\n`], { type: "application/x-ndjson" });
    }
    const token = getToken();
    const response = await fetch(`${API_ROOT}/api/match/${encodeURIComponent(id)}/export`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined
    });
    if (!response.ok) throw new ApiError(`エクスポートに失敗しました (${response.status})`, response.status);
    return response.blob();
  }
};
