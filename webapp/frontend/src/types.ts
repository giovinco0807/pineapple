export type CardCode = string;
export type RowName = "top" | "middle" | "bottom";
export type Street = "T0" | "T1" | "T2" | "T3" | "T4" | "FL";
export type SeatOrder = "first" | "second";

export interface Board {
  top: CardCode[];
  middle: CardCode[];
  bottom: CardCode[];
}

export interface Positions {
  human: SeatOrder;
  ai: SeatOrder;
}

export interface FlStatus {
  human: boolean | number;
  ai: boolean | number;
}

export interface ScoreRow {
  winner: "human" | "ai" | "tie";
  human_rank?: string;
  ai_rank?: string;
  points?: number;
}

export interface HandResult {
  row_wins?: {
    top?: ScoreRow | string;
    middle?: ScoreRow | string;
    bottom?: ScoreRow | string;
  };
  scoop?: "human" | "ai" | null;
  royalties?: {
    human?: number | Record<string, number>;
    ai?: number | Record<string, number>;
  };
  fouls?: { human?: boolean; ai?: boolean };
  fl_entries?: { human?: number | boolean; ai?: number | boolean; cards?: number };
  components?: {
    perspective?: "human";
    foul_base?: number;
    line_total?: number;
    scoop_bonus?: number;
    royalty_delta?: number;
    total?: number;
  };
  raw_score: number;
  capped_score: number;
  stacks_after: { human: number; ai: number };
}

export interface AiTopCandidate {
  rank?: number;
  score?: number | null;
  value?: number | null;
  placements?: Array<[CardCode, RowName]>;
  discards?: CardCode[];
}

export interface AiDecisionMeta {
  evaluator?: string;
  weights_sha?: string[] | string;
  assembly_sha?: string;
  scores_topk?: AiTopCandidate[];
}

export interface ReplayStep {
  id: string;
  index: number;
  actor: "human" | "ai" | "system";
  street: Street | null;
  label: string;
  boards: { human: Board; ai: Board };
  dealt_cards?: CardCode[];
  discards?: CardCode[];
  think_ms?: number;
  ai_meta?: AiDecisionMeta | null;
}

export interface HandSummary {
  id: string;
  index: number;
  status: "playing" | "complete";
  positions?: Positions;
  score?: number;
  capped_score?: number;
  stacks_after?: { human: number; ai: number };
  royalties?: { human: number; ai: number };
  fl_human?: boolean | number;
  fl_ai?: boolean | number;
}

export interface MatchView {
  id: string;
  status: "ready" | "in_hand" | "awaiting_continue" | "completed";
  stacks: { human: number; ai: number };
  first_hand_positions: Positions;
  current_hand_id: string | null;
  can_start_hand: boolean;
  can_continue: boolean;
  hands: HandSummary[];
  assembly_sha?: string;
  created_at?: string;
}

export interface HandView {
  id: string;
  match_id: string;
  index: number;
  status: "playing" | "complete";
  positions: Positions;
  fl_status: FlStatus;
  street: Street | null;
  to_act: "human" | "ai" | null;
  boards: { human: Board; ai: Board };
  dealt_cards: CardCode[];
  action_required: "normal" | "fl" | null;
  result: HandResult | null;
  ai_pending: boolean;
  replay: ReplayStep[];
}

export interface MetaView {
  app_version?: string;
  assembly_sha?: string;
  assembly?: Record<string, unknown>;
  weights?: Record<string, string>;
  rules?: Record<string, unknown>;
}

export interface PlacementPayload {
  placements: Array<[CardCode, RowName]>;
  discards: CardCode[];
}

export interface CreateMatchPayload {
  seed?: number;
}

export const EMPTY_BOARD: Board = {
  top: [],
  middle: [],
  bottom: []
};

export const ROW_CAPACITY: Record<RowName, number> = {
  top: 3,
  middle: 5,
  bottom: 5
};
