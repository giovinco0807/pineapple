import type { DragEvent, MouseEvent } from "react";
import type { CardCode } from "../types";

const SUITS: Record<string, { glyph: string; name: string; red: boolean }> = {
  S: { glyph: "♠", name: "スペード", red: false },
  H: { glyph: "♥", name: "ハート", red: true },
  D: { glyph: "♦", name: "ダイヤ", red: true },
  C: { glyph: "♣", name: "クラブ", red: false },
  "♠": { glyph: "♠", name: "スペード", red: false },
  "♥": { glyph: "♥", name: "ハート", red: true },
  "♦": { glyph: "♦", name: "ダイヤ", red: true },
  "♣": { glyph: "♣", name: "クラブ", red: false }
};

export function describeCard(code: CardCode): {
  rank: string;
  suit: string;
  suitName: string;
  red: boolean;
} {
  const clean = String(code).trim();
  const suitKey = clean.slice(-1).toUpperCase();
  const suit = SUITS[suitKey] ?? { glyph: "•", name: "不明", red: false };
  let rank = clean.slice(0, -1).toUpperCase();
  if (rank === "T") rank = "10";
  return { rank: rank || "?", suit: suit.glyph, suitName: suit.name, red: suit.red };
}

interface CardViewProps {
  card: CardCode;
  selected?: boolean;
  compact?: boolean;
  disabled?: boolean;
  draggableCard?: boolean;
  label?: string;
  onClick?: (card: CardCode) => void;
}

export function CardView({
  card,
  selected = false,
  compact = false,
  disabled = false,
  draggableCard = false,
  label,
  onClick
}: CardViewProps) {
  const face = describeCard(card);

  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    if (!disabled) onClick?.(card);
  };

  const handleDragStart = (event: DragEvent<HTMLButtonElement>) => {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/ofc-card", card);
    event.dataTransfer.setData("text/plain", card);
  };

  return (
    <button
      type="button"
      className={`playing-card ${compact ? "playing-card--compact" : ""} ${
        face.red ? "playing-card--red" : ""
      } ${selected ? "playing-card--selected" : ""} ${disabled ? "playing-card--disabled" : ""}`}
      aria-label={label ?? `${face.suitName}の${face.rank}`}
      aria-pressed={selected}
      draggable={draggableCard && !disabled}
      disabled={disabled}
      onClick={handleClick}
      onDragStart={handleDragStart}
      data-card={card}
    >
      <span className="playing-card__rank">{face.rank}</span>
      <span className="playing-card__suit" aria-hidden="true">
        {face.suit}
      </span>
    </button>
  );
}

export function CardBack({ compact = false, label = "非公開カード" }: { compact?: boolean; label?: string }) {
  return (
    <span
      className={`playing-card playing-card--back ${compact ? "playing-card--compact" : ""}`}
      aria-label={label}
      role="img"
    >
      <span aria-hidden="true">♠</span>
    </span>
  );
}
