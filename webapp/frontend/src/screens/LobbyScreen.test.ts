import { describe, expect, it, vi } from "vitest";
import type { HandView, MatchView } from "../types";
import { matchResumePath } from "./LobbyScreen";

function match(overrides: Partial<MatchView> = {}): MatchView {
  return {
    id: "match-1",
    status: "ready",
    stacks: { human: 200, ai: 200 },
    first_hand_positions: { human: "first", ai: "second" },
    current_hand_id: null,
    can_start_hand: true,
    can_continue: false,
    hands: [],
    ...overrides
  };
}

function unusedClient() {
  return {
    getHand: vi.fn<() => Promise<HandView>>(),
    startHand: vi.fn<() => Promise<HandView>>()
  };
}

describe("matchResumePath", () => {
  it("awaiting_continueでは最新結果へ進み、新しいhandを作らない", async () => {
    const client = unusedClient();
    const path = await matchResumePath(
      match({
        status: "awaiting_continue",
        can_start_hand: false,
        can_continue: true,
        hands: [
          { id: "hand-1", index: 1, status: "complete" },
          { id: "hand-2", index: 2, status: "complete" }
        ]
      }),
      client
    );

    expect(path).toBe("/result/match-1/hand-2");
    expect(client.getHand).not.toHaveBeenCalled();
    expect(client.startHand).not.toHaveBeenCalled();
  });

  it("completedではマッチサマリーへ進み、新しいhandを作らない", async () => {
    const client = unusedClient();
    const path = await matchResumePath(
      match({
        status: "completed",
        can_start_hand: false,
        hands: [{ id: "hand-3", index: 3, status: "complete" }]
      }),
      client
    );

    expect(path).toBe("/summary/match-1");
    expect(client.getHand).not.toHaveBeenCalled();
    expect(client.startHand).not.toHaveBeenCalled();
  });
});
