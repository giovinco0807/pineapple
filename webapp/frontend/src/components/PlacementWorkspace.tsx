import { useEffect, useMemo, useState } from "react";
import { BoardView } from "./BoardView";
import { CardBack, CardView } from "./CardView";
import {
  ROW_CAPACITY,
  type Board,
  type CardCode,
  type PlacementPayload,
  type RowName,
  type Street
} from "../types";

type DraftEntry =
  | { kind: "placement"; card: CardCode; row: RowName }
  | { kind: "discard"; card: CardCode };

interface PlacementWorkspaceProps {
  baseBoard: Board;
  cards: CardCode[];
  street: Street;
  fantasyLand?: boolean;
  busy?: boolean;
  onSubmit: (payload: PlacementPayload) => Promise<void> | void;
}

function boardWithDraft(base: Board, entries: DraftEntry[]): Board {
  return {
    top: [...base.top, ...entries.filter((entry): entry is Extract<DraftEntry, { kind: "placement" }> => entry.kind === "placement" && entry.row === "top").map((entry) => entry.card)],
    middle: [...base.middle, ...entries.filter((entry): entry is Extract<DraftEntry, { kind: "placement" }> => entry.kind === "placement" && entry.row === "middle").map((entry) => entry.card)],
    bottom: [...base.bottom, ...entries.filter((entry): entry is Extract<DraftEntry, { kind: "placement" }> => entry.kind === "placement" && entry.row === "bottom").map((entry) => entry.card)]
  };
}

export function PlacementWorkspace({
  baseBoard,
  cards,
  street,
  fantasyLand = false,
  busy = false,
  onSubmit
}: PlacementWorkspaceProps) {
  const [entries, setEntries] = useState<DraftEntry[]>([]);
  const [selected, setSelected] = useState<CardCode | null>(null);
  const cardsKey = cards.join("|");

  useEffect(() => {
    setEntries([]);
    setSelected(null);
  }, [cardsKey, street]);

  const assigned = useMemo(() => new Set(entries.map((entry) => entry.card)), [entries]);
  const placementEntries = entries.filter(
    (entry): entry is Extract<DraftEntry, { kind: "placement" }> => entry.kind === "placement"
  );
  const discardEntries = entries.filter(
    (entry): entry is Extract<DraftEntry, { kind: "discard" }> => entry.kind === "discard"
  );
  const currentBoard = boardWithDraft(baseBoard, entries);
  const placementTarget = fantasyLand ? 13 : street === "T0" ? cards.length : 2;
  const discardTarget = cards.length - placementTarget;
  const complete = placementEntries.length === placementTarget && discardEntries.length === discardTarget;
  const draftCards = new Set(placementEntries.map((entry) => entry.card));

  const removeCard = (card: CardCode) => {
    setEntries((current) => current.filter((entry) => entry.card !== card));
    setSelected(null);
  };

  const assignRow = (card: CardCode, row: RowName) => {
    if (!cards.includes(card)) return;
    const withoutCard = entries.filter((entry) => entry.card !== card);
    const occupied =
      baseBoard[row].length +
      withoutCard.filter((entry) => entry.kind === "placement" && entry.row === row).length;
    if (occupied >= ROW_CAPACITY[row]) return;
    setEntries([...withoutCard, { kind: "placement", card, row }]);
    setSelected(null);
  };

  const assignSelectedRow = (row: RowName) => {
    if (selected) assignRow(selected, row);
  };

  const assignDiscard = (card: CardCode) => {
    if (!cards.includes(card)) return;
    const withoutCard = entries.filter((entry) => entry.card !== card);
    if (withoutCard.filter((entry) => entry.kind === "discard").length >= discardTarget) return;
    setEntries([...withoutCard, { kind: "discard", card }]);
    setSelected(null);
  };

  const handleSubmit = async () => {
    if (!complete || busy) return;
    await onSubmit({
      placements: placementEntries.map((entry) => [entry.card, entry.row]),
      discards: discardEntries.map((entry) => entry.card)
    });
  };

  return (
    <div className="placement-workspace">
      <BoardView
        board={currentBoard}
        interactive={!busy}
        selectedCard={selected}
        onChooseRow={assignSelectedRow}
        onDropCard={assignRow}
        onRemoveCard={removeCard}
        draftCards={draftCards}
      />

      <section className="dealt-panel" aria-label={fantasyLand ? "Fantasy Landの配札" : "今回の配札"}>
        <div className="section-label">
          <span>{fantasyLand ? `配札 ${cards.length}枚` : `${street}の配札`}</span>
          <small>
            {selected ? `${selected}を選択中` : fantasyLand ? "タップまたはドラッグで配置" : "カードを選び、行をタップ"}
          </small>
        </div>
        <div className={`dealt-cards ${fantasyLand ? "dealt-cards--fl" : ""}`}>
          {cards.map((card) => {
            const entry = entries.find((item) => item.card === card);
            return (
              <div className="dealt-card-wrap" key={card}>
                <CardView
                  card={card}
                  selected={selected === card}
                  disabled={Boolean(entry) || busy}
                  draggableCard={!entry && !busy}
                  onClick={() => setSelected((current) => (current === card ? null : card))}
                />
                {entry ? (
                  <button type="button" className="assignment-badge" onClick={() => removeCard(card)}>
                    {entry.kind === "discard"
                      ? "捨て札"
                      : entry.row === "top"
                        ? "TOP"
                        : entry.row === "middle"
                          ? "MID"
                          : "BTM"}
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>

      {discardTarget > 0 ? (
        <button
          type="button"
          className={`discard-zone ${selected && discardEntries.length < discardTarget ? "discard-zone--ready" : ""}`}
          onClick={() => selected && assignDiscard(selected)}
          onDragOver={(event) => {
            if (discardEntries.length < discardTarget) event.preventDefault();
          }}
          onDrop={(event) => {
            event.preventDefault();
            const card = event.dataTransfer.getData("text/ofc-card") || event.dataTransfer.getData("text/plain");
            if (card) assignDiscard(card);
          }}
          disabled={busy}
          aria-label={`捨て札 ${discardEntries.length}/${discardTarget}枚`}
        >
          <span className="discard-zone__backs">
            {Array.from({ length: discardTarget }, (_, index) =>
              discardEntries[index] ? (
                <CardBack compact key={discardEntries[index].card} label="選択済みの裏向き捨て札" />
              ) : (
                <span className="discard-slot" key={`discard-${index}`}>
                  {selected && index === discardEntries.length ? "+" : ""}
                </span>
              )
            )}
          </span>
          <span>
            <strong>裏向き捨て札</strong>
            <small>
              {discardEntries.length}/{discardTarget}枚 · 確定まで相手には見えません
            </small>
          </span>
        </button>
      ) : null}

      <div className="draft-toolbar">
        <button
          type="button"
          className="button button--ghost"
          onClick={() => {
            setEntries((current) => current.slice(0, -1));
            setSelected(null);
          }}
          disabled={entries.length === 0 || busy}
        >
          ↶ 1手戻す
        </button>
        <span>
          配置 {placementEntries.length}/{placementTarget}
          {discardTarget ? ` · 捨て ${discardEntries.length}/${discardTarget}` : ""}
        </span>
        <button type="button" className="button button--primary" disabled={!complete || busy} onClick={handleSubmit}>
          {busy ? "送信中…" : "この配置で確定"}
        </button>
      </div>
      <p className="server-note">確定後の取り消しはできません。最終的な合法性はサーバーが検証します。</p>
    </div>
  );
}
