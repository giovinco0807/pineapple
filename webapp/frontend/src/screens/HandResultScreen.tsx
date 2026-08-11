import { useState } from "react";
import { useNavigate, useParams } from "../router";
import { api } from "../api";
import { AppShell, ErrorState, LoadingState, ScoreRail } from "../components/AppShell";
import { BoardView } from "../components/BoardView";
import { errorMessage, handPath, useHand, useMatch } from "../hooks";
import type { HandResult, RowName, ScoreRow } from "../types";

const ROW_LABELS: Record<RowName, string> = {
  top: "トップ",
  middle: "ミドル",
  bottom: "ボトム"
};

export function royaltyTotal(value: number | Record<string, number> | undefined): number {
  if (typeof value === "number") return value;
  if (!value) return 0;
  const declaredTotal = Number(value.total);
  if (Number.isFinite(declaredTotal)) return declaredTotal;
  return (["top", "middle", "bottom"] as const).reduce(
    (sum, row) => sum + (Number.isFinite(Number(value[row])) ? Number(value[row]) : 0),
    0
  );
}

function scoreRow(result: HandResult, row: RowName): ScoreRow {
  const raw = result.row_wins?.[row];
  if (raw && typeof raw === "object") return raw;
  const winner =
    typeof raw === "string" && raw.toLowerCase().includes("human")
      ? "human"
      : typeof raw === "string" && raw.toLowerCase().includes("ai")
        ? "ai"
        : "tie";
  return { winner, points: winner === "human" ? 1 : winner === "ai" ? -1 : 0 };
}

function signed(points: number): string {
  return `${points > 0 ? "+" : ""}${points}`;
}

function ComponentLine({
  label,
  detail,
  points,
  tone
}: {
  label: string;
  detail: string;
  points: number;
  tone?: "foul" | "scoop";
}) {
  return (
    <div className={`score-component ${tone ? `score-component--${tone}` : ""}`}>
      <span>
        <strong>{label}</strong>
        <small>{detail}</small>
      </span>
      <em>{signed(points)}</em>
    </div>
  );
}

export function ResultBreakdown({ result }: { result: HandResult }) {
  const humanRoyalty = royaltyTotal(result.royalties?.human);
  const aiRoyalty = royaltyTotal(result.royalties?.ai);
  const hasFoul = Boolean(result.fouls?.human || result.fouls?.ai);
  const components = result.components;
  const fallbackScoop =
    result.scoop === "human" ? 3 : result.scoop === "ai" ? -3 : 0;
  const scoopBonus = components?.scoop_bonus ?? fallbackScoop;
  const royaltyDelta = components?.royalty_delta ?? humanRoyalty - aiRoyalty;

  return (
    <section className="result-breakdown">
      <div className="section-label">
        <span>スコア内訳</span>
        <small>サーバーの公式採点</small>
      </div>
      {(["top", "middle", "bottom"] as RowName[]).map((row) => {
        const detail = scoreRow(result, row);
        return (
          <div className="result-row" key={row}>
            <span
              className={`result-row__outcome result-row__outcome--${hasFoul ? "foul" : detail.winner}`}
            >
              {hasFoul
                ? "FOUL"
                : detail.winner === "human"
                  ? "WIN"
                  : detail.winner === "ai"
                    ? "LOSE"
                    : "PUSH"}
            </span>
            <span>
              <strong>{ROW_LABELS[row]}</strong>
              <small>
                {hasFoul
                  ? "ファウル精算では行ごとの±1を加算しません"
                  : `${detail.human_rank ?? "あなた"} vs ${detail.ai_rank ?? "AI"}`}
              </small>
            </span>
            <em>
              {hasFoul
                ? "—"
                : detail.points == null
                  ? "0"
                  : signed(detail.points)}
            </em>
          </div>
        );
      })}

      <div className="score-components" aria-label="公式スコア構成">
        {hasFoul ? (
          components?.foul_base != null ? (
            <ComponentLine
              label="ファウル"
              detail={result.fouls?.human ? "あなたの盤面" : "AIの盤面"}
              points={components.foul_base}
              tone="foul"
            />
          ) : null
        ) : (
          <>
            {components?.line_total != null ? (
              <ComponentLine
                label="行ポイント"
                detail="トップ・ミドル・ボトム合計"
                points={components.line_total}
              />
            ) : null}
            {scoopBonus !== 0 ? (
              <ComponentLine
                label="スクープボーナス"
                detail={scoopBonus > 0 ? "YOU · 3行すべて勝利" : "AI · 3行すべて勝利"}
                points={scoopBonus}
                tone="scoop"
              />
            ) : null}
          </>
        )}
        {components?.royalty_delta != null ? (
          <ComponentLine
            label="ロイヤリティ差"
            detail={`YOU ${signed(humanRoyalty)} / AI ${signed(aiRoyalty)}`}
            points={royaltyDelta}
          />
        ) : null}
      </div>

      <div className="royalty-line">
        <span>
          ロイヤリティ <strong>YOU {signed(humanRoyalty)}</strong>
        </span>
        <span>
          <strong>AI {signed(aiRoyalty)}</strong>
        </span>
      </div>
      <div className="official-total">
        <span>公式生スコア</span>
        <strong>{signed(result.raw_score)}</strong>
      </div>
      <p className="result-authority-note">
        {hasFoul
          ? "ファウル処理とロイヤリティ差を含む公式値です。"
          : "行ポイント・スクープ・ロイヤリティ差をRust score_finalで合算した公式値です。"}
      </p>
    </section>
  );
}

export function FantasyLandEarned({
  entries
}: {
  entries: HandResult["fl_entries"];
}) {
  if (!entries?.human && !entries?.ai) return null;
  const cards = Number.isFinite(Number(entries.cards)) ? Number(entries.cards) : 14;
  return (
    <div className="fl-earned">
      <span aria-hidden="true">✦</span>
      <p>
        <strong>次はFantasy Land</strong>
        <span className="fl-earned__entries">
          {entries.human ? <span>あなた {cards}枚</span> : null}
          {entries.ai ? <span>AI {cards}枚</span> : null}
        </span>
      </p>
    </div>
  );
}

export function HandResultScreen() {
  const { matchId = "", handId = "" } = useParams();
  const navigate = useNavigate();
  const { hand, loading, error, reload } = useHand(handId);
  const { match, setMatch } = useMatch(matchId);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const advance = async (shouldContinue: boolean) => {
    setBusy(true);
    setActionError(null);
    try {
      const nextMatch = match?.can_continue
        ? await api.continueMatch(matchId, shouldContinue)
        : match ?? (await api.getMatch(matchId));
      setMatch(nextMatch);
      if (!shouldContinue || nextMatch.status === "completed") {
        navigate(`/summary/${matchId}`);
        return;
      }
      if (nextMatch.current_hand_id && nextMatch.current_hand_id !== handId) {
        navigate(handPath(await api.getHand(nextMatch.current_hand_id)));
        return;
      }
      navigate(handPath(await api.startHand(matchId)));
    } catch (caught) {
      setActionError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const result = hand?.result;
  const transfer = result?.capped_score ?? 0;
  const humanWon = transfer > 0;

  return (
    <AppShell matchId={matchId} compactHeader>
      {match ? <ScoreRail human={match.stacks.human} ai={match.stacks.ai} handIndex={hand?.index} /> : null}
      {loading && !hand ? <LoadingState label="採点結果を読み込んでいます" /> : null}
      {error && !hand ? <ErrorState message={error} onRetry={() => void reload()} /> : null}
      {hand && result ? (
        <>
          <section className={`result-hero ${transfer < 0 ? "result-hero--loss" : ""}`}>
            <p>HAND {hand.index} RESULT</p>
            <h1>{transfer === 0 ? "引き分け" : humanWon ? "YOU WIN" : "AI WIN"}</h1>
            <div className="score-transfer" aria-label={`点数移動 ${transfer >= 0 ? "+" : ""}${transfer}`}>
              <span className="score-transfer__coin" aria-hidden="true">
                P
              </span>
              <strong>
                {transfer >= 0 ? "+" : ""}
                {transfer}
              </strong>
              <small>points</small>
            </div>
            {result.raw_score !== result.capped_score ? (
              <p>生スコア {result.raw_score > 0 ? "+" : ""}{result.raw_score} → 残点上限で精算</p>
            ) : null}
          </section>

          <ResultBreakdown result={result} />

          <details className="final-boards">
            <summary>完成盤面を見る</summary>
            <BoardView board={hand.boards.ai} title="AI" compact />
            <BoardView board={hand.boards.human} title="あなた" compact />
          </details>

          <FantasyLandEarned entries={result.fl_entries} />

          {actionError ? <p className="inline-error" role="alert">{actionError}</p> : null}
          <div className="result-actions">
            {match?.status === "completed" ? (
              <button type="button" className="button button--primary button--large" onClick={() => navigate(`/summary/${matchId}`)}>
                マッチ結果へ
              </button>
            ) : match?.can_continue ? (
              <>
                <button type="button" className="button button--primary button--large" disabled={busy} onClick={() => advance(true)}>
                  {busy ? "準備中…" : "続行する"}
                </button>
                <button type="button" className="button button--ghost" disabled={busy} onClick={() => advance(false)}>
                  ここで終了
                </button>
              </>
            ) : (
              <button type="button" className="button button--primary button--large" disabled={busy} onClick={() => advance(true)}>
                {busy ? "FLを準備中…" : "次のハンドへ"}
              </button>
            )}
          </div>
        </>
      ) : hand && !result ? (
        <ErrorState message="採点結果がまだ確定していません" onRetry={() => void reload()} />
      ) : null}
    </AppShell>
  );
}
