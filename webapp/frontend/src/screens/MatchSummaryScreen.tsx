import { useNavigate, useParams } from "../router";
import { AppShell, ErrorState, LoadingState, ScoreRail, ScreenHeading } from "../components/AppShell";
import { useMatch } from "../hooks";
import type { HandSummary } from "../types";

export function sumMatchRoyalties(hands: HandSummary[]): { human: number; ai: number } {
  return hands.reduce(
    (total, hand) => ({
      human: total.human + (hand.royalties?.human ?? 0),
      ai: total.ai + (hand.royalties?.ai ?? 0)
    }),
    { human: 0, ai: 0 }
  );
}

export function RoyaltySummaryCards({
  totals
}: {
  totals: { human: number; ai: number };
}) {
  return (
    <>
      <article>
        <small>ロイヤリティ合計</small>
        <strong>+{totals.human}</strong>
        <span>YOU</span>
      </article>
      <article>
        <small>ロイヤリティ合計</small>
        <strong>+{totals.ai}</strong>
        <span>AI</span>
      </article>
    </>
  );
}

export function MatchSummaryScreen() {
  const { matchId = "" } = useParams();
  const navigate = useNavigate();
  const { match, loading, error, reload } = useMatch(matchId);

  const humanWins = match?.hands.filter((hand) => (hand.capped_score ?? hand.score ?? 0) > 0).length ?? 0;
  const aiWins = match?.hands.filter((hand) => (hand.capped_score ?? hand.score ?? 0) < 0).length ?? 0;
  const humanFl =
    match?.hands.reduce((count, hand) => count + (hand.fl_human ? 1 : 0), 0) ?? 0;
  const aiFl = match?.hands.reduce((count, hand) => count + (hand.fl_ai ? 1 : 0), 0) ?? 0;
  const royaltyTotals = sumMatchRoyalties(match?.hands ?? []);
  const won = (match?.stacks.human ?? 0) > (match?.stacks.ai ?? 0);

  return (
    <AppShell matchId={matchId}>
      {loading && !match ? <LoadingState label="マッチ結果を集計しています" /> : null}
      {error && !match ? <ErrorState message={error} onRetry={() => void reload()} /> : null}
      {match ? (
        <>
          <section className={`summary-hero ${won ? "" : "summary-hero--loss"}`}>
            <p>MATCH COMPLETE</p>
            <h1>{match.stacks.human === match.stacks.ai ? "DRAW" : won ? "VICTORY" : "DEFEAT"}</h1>
            <ScoreRail human={match.stacks.human} ai={match.stacks.ai} />
            <div className="summary-delta">
              <span>
                YOU <strong>{match.stacks.human - 200 >= 0 ? "+" : ""}{match.stacks.human - 200}</strong>
              </span>
              <span>
                AI <strong>{match.stacks.ai - 200 >= 0 ? "+" : ""}{match.stacks.ai - 200}</strong>
              </span>
            </div>
          </section>

          <section className="summary-panel">
            <ScreenHeading eyebrow="OVERVIEW" title={`${match.hands.length}ハンドの記録`} />
            <div className="stat-grid">
              <article>
                <small>ハンド勝利</small>
                <strong>{humanWins}</strong>
                <span>YOU</span>
              </article>
              <article>
                <small>ハンド勝利</small>
                <strong>{aiWins}</strong>
                <span>AI</span>
              </article>
              <article>
                <small>Fantasy Land</small>
                <strong>{humanFl}</strong>
                <span>YOU</span>
              </article>
              <article>
                <small>Fantasy Land</small>
                <strong>{aiFl}</strong>
                <span>AI</span>
              </article>
              <RoyaltySummaryCards totals={royaltyTotals} />
            </div>

            <div className="score-history" aria-label="ハンドごとの点数移動">
              <div className="section-label">
                <span>点数推移</span>
                <small>あなた視点</small>
              </div>
              <div className="score-history__bars">
                {match.hands.map((hand) => {
                  const score = hand.capped_score ?? hand.score ?? 0;
                  return (
                    <div key={hand.id} className="score-history__item">
                      <span className={score < 0 ? "negative" : ""} style={{ height: `${Math.min(50, Math.abs(score) * 2 + 7)}px` }} />
                      <small>H{hand.index}</small>
                      <em>{score > 0 ? "+" : ""}{score}</em>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <div className="summary-actions">
            <button type="button" className="button button--primary" onClick={() => navigate(`/history/${matchId}`)}>
              ハンドを振り返る
            </button>
            <button type="button" className="button button--secondary" onClick={() => navigate(`/export/${matchId}`)}>
              JSONLを出力
            </button>
            <button type="button" className="button button--ghost" onClick={() => navigate("/")}>
              ロビーに戻る
            </button>
          </div>
        </>
      ) : null}
    </AppShell>
  );
}
