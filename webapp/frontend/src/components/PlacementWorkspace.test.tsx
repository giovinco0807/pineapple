import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PlacementWorkspace } from "./PlacementWorkspace";
import { EMPTY_BOARD } from "../types";

describe("PlacementWorkspace", () => {
  it("カードを行へ仮配置し、確定前に1手戻せる", async () => {
    const user = userEvent.setup();
    render(
      <PlacementWorkspace
        baseBoard={EMPTY_BOARD}
        cards={["AS", "KH", "QD", "JC", "9S"]}
        street="T0"
        onSubmit={vi.fn()}
      />
    );

    await user.click(screen.getByRole("button", { name: "スペードのA" }));
    await user.click(screen.getByRole("button", { name: /トップ、0\/3枚/ }));
    expect(screen.getByRole("button", { name: "ASの仮配置を取り消す" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "↶ 1手戻す" }));
    expect(screen.queryByRole("button", { name: "ASの仮配置を取り消す" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "この配置で確定" })).toBeDisabled();
  });

  it("T1は2枚配置と1枚捨て札が揃うまで送信しない", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <PlacementWorkspace
        baseBoard={EMPTY_BOARD}
        cards={["AS", "KH", "QD"]}
        street="T1"
        onSubmit={onSubmit}
      />
    );

    await user.click(screen.getByRole("button", { name: "スペードのA" }));
    await user.click(screen.getByRole("button", { name: /トップ、0\/3枚/ }));
    await user.click(screen.getByRole("button", { name: "ハートのK" }));
    await user.click(screen.getByRole("button", { name: /ミドル、0\/5枚/ }));
    expect(screen.getByRole("button", { name: "この配置で確定" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "ダイヤのQ" }));
    await user.click(screen.getByRole("button", { name: /捨て札 0\/1枚/ }));
    await user.click(screen.getByRole("button", { name: "この配置で確定" }));

    expect(onSubmit).toHaveBeenCalledWith({
      placements: [
        ["AS", "top"],
        ["KH", "middle"]
      ],
      discards: ["QD"]
    });
  });
});
