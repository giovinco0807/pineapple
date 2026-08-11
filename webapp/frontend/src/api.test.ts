import { afterEach, describe, expect, it, vi } from "vitest";
import { api, normalizeHand, setToken } from "./api";

describe("API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    setToken("");
  });

  it("localStorageのBearer tokenをリクエストへ付与する", async () => {
    setToken("shared-secret");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ app_version: "test" }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getMeta()).resolves.toMatchObject({ app_version: "test" });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer shared-secret");
  });

  it("相手の秘密手札やデッキ順をHandViewへ取り込まない", () => {
    const hand = normalizeHand({
      hand_id: "h1",
      match_id: "m1",
      status: "playing",
      positions: { human: "first", ai: "second" },
      street: "T1",
      to_act: "human",
      action_required: "normal",
      boards: {
        human: { top: [], middle: [], bottom: [] },
        ai: { top: ["AS"], middle: [], bottom: [] }
      },
      dealt_cards: ["KH", "QD", "JC"],
      opponent_hand: ["2S", "3S", "4S"],
      deck_order: ["5S", "6S"]
    });

    expect(hand.boards.ai.top).toEqual(["AS"]);
    expect(hand.dealt_cards).toEqual(["KH", "QD", "JC"]);
    expect(hand).not.toHaveProperty("opponent_hand");
    expect(hand).not.toHaveProperty("deck_order");
  });

  it("採点は先攻視点より人間視点を優先し、AIメタの配列SHAとnullスコアを保持する", () => {
    const hand = normalizeHand({
      hand_id: "h2",
      match_id: "m1",
      status: "complete",
      positions: { human: "second", ai: "first" },
      boards: { human: {}, ai: {} },
      result: {
        raw_score: 12,
        capped_score: 12,
        human_raw_score: -12,
        human_capped_score: -12,
        point_components: {
          perspective: "human",
          foul_base: -6,
          line_total: 0,
          scoop_bonus: 0,
          royalty_delta: -6,
          total: -12
        },
        stacks_after: { human: 188, ai: 212 }
      },
      replay: [
        {
          actor: "ai",
          street: "T4",
          boards: { human: {}, ai: {} },
          ai_meta: {
            evaluator: "exact",
            weights_sha: ["sha-a", "sha-b"],
            scores_topk: [{ rank: 1, score: null }]
          }
        }
      ]
    });

    expect(hand.result?.capped_score).toBe(-12);
    expect(hand.result?.components).toEqual({
      perspective: "human",
      foul_base: -6,
      line_total: 0,
      scoop_bonus: 0,
      royalty_delta: -6,
      total: -12
    });
    expect(hand.replay[0].ai_meta?.weights_sha).toEqual(["sha-a", "sha-b"]);
    expect(hand.replay[0].ai_meta?.scores_topk?.[0].score).toBeNull();
  });

  it("デモ盤面のAIファウルを6点と有効側ロイヤリティで精算する", async () => {
    setToken("demo");
    const match = await api.createMatch();
    let hand = await api.startHand(match.id);

    for (let street = 0; street < 5; street += 1) {
      hand = await api.submitAction(hand.id, {
        placements: [],
        discards: []
      });
    }

    expect(hand.boards.ai.middle).toEqual(["8S", "7S", "6S", "5S", "3S"]);
    expect(hand.boards.ai.bottom).toEqual(["10S", "10D", "9H", "9C", "6D"]);
    expect(hand.result).toMatchObject({
      fouls: { human: false, ai: true },
      row_wins: {
        top: "not_scored_foul",
        middle: "not_scored_foul",
        bottom: "not_scored_foul"
      },
      royalties: { human: 8, ai: 0 },
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
    });
  });
});
