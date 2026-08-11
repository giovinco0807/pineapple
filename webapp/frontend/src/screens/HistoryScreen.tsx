import { useEffect, useMemo, useState } from "react";
import { useParams } from "../router";
import { api } from "../api";
import { AppShell, ErrorState, LoadingState, ScreenHeading } from "../components/AppShell";
import { BoardView } from "../components/BoardView";
import { errorMessage, useMatch } from "../hooks";
import type { HandView, ReplayStep } from "../types";

function finalStep(hand: HandView): ReplayStep {
  return {
    id: `${hand.id}-final`,
    index: hand.replay.length,
    actor: "system",
    street: null,
    label: hand.status === "complete" ? "最終盤面" : "現在の公開状態",
    boards: hand.boards
  };
}

export function HistoryScreen() {
  const { matchId = "" } = useParams();
  const { match, loading, error, reload } = useMatch(matchId);
  const [selectedHandId, setSelectedHandId] = useState<string | null>(null);
  const [hand, setHand] = useState<HandView | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [handLoading, setHandLoading] = useState(false);
  const [handError, setHandError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedHandId && match?.hands.length) {
      setSelectedHandId(match.hands[match.hands.length - 1].id);
    }
  }, [match, selectedHandId]);

  useEffect(() => {
    if (!selectedHandId) return;
    let active = true;
    setHandLoading(true);
    setHandError(null);
    api
      .getHand(selectedHandId)
      .then((next) => {
        if (!active) return;
        setHand(next);
        setStepIndex(0);
      })
      .catch((caught) => active && setHandError(errorMessage(caught)))
      .finally(() => active && setHandLoading(false));
    return () => {
      active = false;
    };
  }, [selectedHandId]);

  const steps = useMemo(() => (hand ? [...hand.replay, finalStep(hand)] : []), [hand]);
  const step = steps[Math.min(stepIndex, Math.max(steps.length - 1, 0))];

  return (
    <AppShell matchId={matchId}>
      <ScreenHeading eyebrow="REPLAY" title="一手ずつ振り返る">
        <p>確定済みの記録だけを再生します。AIの非公開情報は対局終了後の記録に限ります。</p>
      </ScreenHeading>
      {loading && !match ? <LoadingState label="履歴を読み込んでいます" /> : null}
      {error && !match ? <ErrorState message={error} onRetry={() => void reload()} /> : null}
      {match ? (
        <div className="hand-tabs" role="tablist" aria-label="ハンドを選択">
          {match.hands.map((summary) => (
            <button
              type="button"
              role="tab"
              aria-selected={selectedHandId === summary.id}
              className={selectedHandId === summary.id ? "active" : ""}
              key={summary.id}
              onClick={() => setSelectedHandId(summary.id)}
            >
              <span>H{summary.index}</span>
              <small>
                {(summary.capped_score ?? summary.score ?? 0) > 0 ? "+" : ""}
                {summary.capped_score ?? summary.score ?? 0}
              </small>
            </button>
          ))}
        </div>
      ) : null}

      {handLoading && !hand ? <LoadingState label="ハンド記録を展開しています" /> : null}
      {handError ? <ErrorState message={handError} /> : null}
      {hand && step ? (
        <>
          <section className="replay-status">
            <span>
              {step.street ?? "END"} · {step.actor === "human" ? "YOU" : step.actor === "ai" ? "AI" : "RESULT"}
            </span>
            <strong>{step.label}</strong>
            <small>
              {step.index + 1}/{steps.length}
            </small>
          </section>
          <BoardView board={step.boards.ai} title="AI" compact />
          <BoardView board={step.boards.human} title="あなた" compact />

          {step.ai_meta ? (
            <details className="ai-meta">
              <summary>AI評価を見る</summary>
              <dl>
                <div>
                  <dt>Evaluator</dt>
                  <dd>{step.ai_meta.evaluator ?? "—"}</dd>
                </div>
                <div>
                  <dt>Weights SHA</dt>
                  <dd>
                    {Array.isArray(step.ai_meta.weights_sha)
                      ? step.ai_meta.weights_sha.join(", ")
                      : step.ai_meta.weights_sha ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>思考時間</dt>
                  <dd>{step.think_ms != null ? `${step.think_ms} ms` : "—"}</dd>
                </div>
              </dl>
              {step.ai_meta.scores_topk?.length ? (
                <ol className="top-candidates">
                  {step.ai_meta.scores_topk.slice(0, 3).map((candidate, index) => {
                    const score = candidate.score ?? candidate.value ?? null;
                    return (
                    <li key={`${candidate.rank ?? index}-${score ?? "unavailable"}`}>
                      <span>#{candidate.rank ?? index + 1}</span>
                      <strong>{score == null ? "unavailable" : score.toFixed(3)}</strong>
                    </li>
                    );
                  })}
                </ol>
              ) : null}
            </details>
          ) : null}

          <div className="replay-controls">
            <button type="button" className="button button--secondary" disabled={stepIndex <= 0} onClick={() => setStepIndex((value) => value - 1)}>
              ← 前の手
            </button>
            <input
              type="range"
              min="0"
              max={Math.max(steps.length - 1, 0)}
              value={stepIndex}
              aria-label="再生位置"
              onChange={(event) => setStepIndex(Number(event.target.value))}
            />
            <button
              type="button"
              className="button button--primary"
              disabled={stepIndex >= steps.length - 1}
              onClick={() => setStepIndex((value) => value + 1)}
            >
              次の手 →
            </button>
          </div>
        </>
      ) : match && match.hands.length === 0 ? (
        <div className="empty-panel">
          <strong>再生できるハンドがありません</strong>
          <p>ハンドを完了すると、ここに一手ずつ記録されます。</p>
        </div>
      ) : null}
    </AppShell>
  );
}
