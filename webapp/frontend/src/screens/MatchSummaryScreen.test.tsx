import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { normalizeMatch } from "../api";
import {
  RoyaltySummaryCards,
  sumMatchRoyalties
} from "./MatchSummaryScreen";

function completedResult(
  human: Record<string, number>,
  ai: Record<string, number>
) {
  return {
    human_raw_score: 1,
    human_capped_score: 1,
    royalties: { human, ai },
    stacks_after: { human: 201, ai: 199 }
  };
}

describe("MatchSummaryScreen royalties", () => {
  it("複数handをtotal優先・totalなしは3行合計でYOU/AI別に集計表示する", () => {
    const match = normalizeMatch({
      id: "match-royalties",
      status: "completed",
      stacks: { human: 210, ai: 190 },
      first_hand_positions: { human: "first", ai: "second" },
      hands: [
        {
          id: "h1",
          index: 1,
          status: "complete",
          result: completedResult(
            { top: 0, middle: 4, bottom: 0, total: 4 },
            { top: 2, middle: 0, bottom: 0, total: 2 }
          )
        },
        {
          id: "h2",
          index: 2,
          status: "complete",
          result: completedResult(
            { top: 1, middle: 2, bottom: 3 },
            { top: 0, middle: 3, bottom: 0, total: 3 }
          )
        },
        {
          id: "h3",
          index: 3,
          status: "playing",
          result: null
        }
      ]
    });

    expect(match.hands[0].royalties).toEqual({ human: 4, ai: 2 });
    expect(match.hands[1].royalties).toEqual({ human: 6, ai: 3 });
    expect(match.hands[2].royalties).toBeUndefined();

    const totals = sumMatchRoyalties(match.hands);
    expect(totals).toEqual({ human: 10, ai: 5 });

    render(<RoyaltySummaryCards totals={totals} />);
    expect(screen.getByText("+10")).toBeInTheDocument();
    expect(screen.getByText("+5")).toBeInTheDocument();
    expect(screen.getAllByText("ロイヤリティ合計")).toHaveLength(2);
  });
});
