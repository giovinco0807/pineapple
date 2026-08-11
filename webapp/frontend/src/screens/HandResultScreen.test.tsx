import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { HandResult } from "../types";
import {
  FantasyLandEarned,
  ResultBreakdown,
  royaltyTotal
} from "./HandResultScreen";

function result(overrides: Partial<HandResult> = {}): HandResult {
  return {
    row_wins: {
      top: "human",
      middle: "human",
      bottom: "human"
    },
    scoop: "human",
    royalties: {
      human: { top: 0, middle: 4, bottom: 0, total: 4 },
      ai: { top: 0, middle: 0, bottom: 0, total: 0 }
    },
    fouls: { human: false, ai: false },
    components: {
      perspective: "human",
      foul_base: 0,
      line_total: 3,
      scoop_bonus: 3,
      royalty_delta: 4,
      total: 10
    },
    raw_score: 10,
    capped_score: 10,
    stacks_after: { human: 210, ai: 190 },
    ...overrides
  };
}

describe("HandResultScreen score breakdown", () => {
  it("total付きロイヤリティを二重計上せず、スクープと公式合計を表示する", () => {
    expect(royaltyTotal({ top: 0, middle: 4, bottom: 0, total: 4 })).toBe(4);
    render(<ResultBreakdown result={result()} />);

    expect(screen.getByText("YOU +4")).toBeInTheDocument();
    expect(screen.queryByText("YOU +8")).not.toBeInTheDocument();
    expect(screen.getByText("スクープボーナス")).toBeInTheDocument();
    expect(screen.getAllByText("+3")).toHaveLength(2);
    expect(screen.getByText("公式生スコア")).toBeInTheDocument();
    expect(screen.getByText("+10")).toBeInTheDocument();
  });

  it("ファウル時は行±1やスクープへ誤配分せず、公式componentsを表示する", () => {
    render(
      <ResultBreakdown
        result={result({
          row_wins: {
            top: "not_scored_foul",
            middle: "not_scored_foul",
            bottom: "not_scored_foul"
          },
          scoop: null,
          royalties: {
            human: { top: 2, middle: 0, bottom: 0, total: 2 },
            ai: { top: 0, middle: 0, bottom: 0, total: 0 }
          },
          fouls: { human: true, ai: false },
          components: {
            perspective: "human",
            foul_base: -6,
            line_total: 0,
            scoop_bonus: 0,
            royalty_delta: 2,
            total: -4
          },
          raw_score: -4,
          capped_score: -4
        })}
      />
    );

    expect(screen.getByText("ファウル", { selector: ".score-component strong" })).toBeInTheDocument();
    expect(screen.getByText("-6")).toBeInTheDocument();
    expect(screen.getByText("ロイヤリティ差")).toBeInTheDocument();
    expect(screen.getByText("+2")).toBeInTheDocument();
    expect(screen.getByText("-4")).toBeInTheDocument();
    expect(screen.queryByText("スクープボーナス")).not.toBeInTheDocument();
  });

  it("両者のRegular Fantasy Landをtrue枚ではなく14枚で個別表示する", () => {
    render(<FantasyLandEarned entries={{ human: true, ai: true, cards: 14 }} />);

    expect(screen.getByText("あなた 14枚")).toBeInTheDocument();
    expect(screen.getByText("AI 14枚")).toBeInTheDocument();
    expect(screen.queryByText(/true枚/)).not.toBeInTheDocument();
  });
});
