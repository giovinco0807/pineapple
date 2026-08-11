import type { DragEvent, KeyboardEvent } from "react";
import { CardView } from "./CardView";
import { ROW_CAPACITY, type Board, type CardCode, type RowName } from "../types";

const ROWS: Array<{ id: RowName; label: string; short: string }> = [
  { id: "top", label: "トップ", short: "TOP" },
  { id: "middle", label: "ミドル", short: "MID" },
  { id: "bottom", label: "ボトム", short: "BTM" }
];

interface BoardViewProps {
  board: Board;
  title?: string;
  interactive?: boolean;
  selectedCard?: CardCode | null;
  privacyMessage?: string;
  onChooseRow?: (row: RowName) => void;
  onDropCard?: (card: CardCode, row: RowName) => void;
  onRemoveCard?: (card: CardCode) => void;
  draftCards?: Set<CardCode>;
  compact?: boolean;
}

export function BoardView({
  board,
  title,
  interactive = false,
  selectedCard,
  privacyMessage,
  onChooseRow,
  onDropCard,
  onRemoveCard,
  draftCards = new Set(),
  compact = false
}: BoardViewProps) {
  const handleDrop = (event: DragEvent<HTMLElement>, row: RowName) => {
    event.preventDefault();
    const card = event.dataTransfer.getData("text/ofc-card") || event.dataTransfer.getData("text/plain");
    if (card) onDropCard?.(card, row);
  };

  return (
    <section className={`board ${compact ? "board--compact" : ""}`} aria-label={title ?? "盤面"}>
      {title ? <h2 className="board__title">{title}</h2> : null}
      <div className="board__rows">
        {ROWS.map((row) => {
          const rowCards = board[row.id] ?? [];
          const free = Math.max(ROW_CAPACITY[row.id] - rowCards.length, 0);
          const canPlace = interactive && free > 0 && Boolean(selectedCard);
          return (
            <div
              key={row.id}
              data-row={row.id}
              data-testid={`board-row-${row.id}`}
              className={`board-row ${canPlace ? "board-row--ready" : ""}`}
              onClick={() => interactive && onChooseRow?.(row.id)}
              onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
                if (interactive && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onChooseRow?.(row.id);
                }
              }}
              onDragOver={(event) => {
                if (interactive && free > 0) event.preventDefault();
              }}
              onDrop={(event) => handleDrop(event, row.id)}
              role="button"
              tabIndex={interactive ? 0 : -1}
              aria-disabled={!interactive}
              aria-label={`${row.label}、${rowCards.length}/${ROW_CAPACITY[row.id]}枚`}
            >
              <span className="board-row__label">
                <span>{row.label}</span>
                <small>
                  {rowCards.length}/{ROW_CAPACITY[row.id]}
                </small>
              </span>
              <span className="board-row__cards">
                {rowCards.map((card) => (
                  <CardView
                    card={card}
                    key={card}
                    compact={compact}
                    label={
                      draftCards.has(card)
                        ? `${card}の仮配置を取り消す`
                        : `${card}、${row.label}に確定済み`
                    }
                    onClick={draftCards.has(card) ? onRemoveCard : undefined}
                    disabled={!draftCards.has(card)}
                  />
                ))}
                {Array.from({ length: free }, (_, slot) => (
                  <span
                    className={`card-slot ${canPlace ? "card-slot--ready" : ""}`}
                    key={`${row.id}-${slot}`}
                    aria-hidden="true"
                  >
                    {canPlace && slot === 0 ? "+" : ""}
                  </span>
                ))}
              </span>
            </div>
          );
        })}
      </div>
      {privacyMessage ? (
        <div className="privacy-strip">
          <span aria-hidden="true">◈</span>
          <span>{privacyMessage}</span>
        </div>
      ) : null}
    </section>
  );
}
