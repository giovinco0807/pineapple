import { useEffect, useState } from "react";
import { useNavigate, useParams } from "../router";
import { api } from "../api";
import { AppShell, ErrorState, LoadingState, ScoreRail } from "../components/AppShell";
import { BoardView } from "../components/BoardView";
import { PlacementWorkspace } from "../components/PlacementWorkspace";
import { errorMessage, handPath, useHand, useMatch } from "../hooks";
import type { PlacementPayload } from "../types";

export function GameScreen() {
  const { matchId = "", handId = "" } = useParams();
  const navigate = useNavigate();
  const { hand, setHand, loading, error, reload } = useHand(handId, true);
  const { match } = useMatch(matchId);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    if (!hand) return;
    if (hand.status === "complete" || (hand.action_required === "fl" && hand.to_act === "human")) {
      navigate(handPath(hand), { replace: true });
    }
  }, [hand, navigate]);

  const submit = async (payload: PlacementPayload) => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const next = await api.submitAction(handId, payload);
      setHand(next);
      if (next.status === "complete" || next.action_required === "fl") navigate(handPath(next), { replace: true });
    } catch (caught) {
      setSubmitError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AppShell matchId={matchId} compactHeader>
      {match ? <ScoreRail human={match.stacks.human} ai={match.stacks.ai} handIndex={hand?.index} /> : null}
      {loading && !hand ? <LoadingState label="盤面を読み込んでいます" /> : null}
      {error && !hand ? <ErrorState message={error} onRetry={() => void reload()} /> : null}
      {hand ? (
        <>
          <div className="turn-banner">
            <span>{hand.street ?? "—"}</span>
            <strong>
              {hand.to_act === "human"
                ? "あなたの手番"
                : hand.to_act === "ai"
                  ? "AIが考えています"
                  : "配置を確認中"}
            </strong>
            <small>{hand.positions.human === "first" ? "あなたが先攻" : "あなたが後攻"}</small>
          </div>

          <BoardView
            board={hand.boards.ai}
            title="AI · 公開盤面"
            compact
            privacyMessage={
              hand.fl_status.ai && hand.boards.ai.top.length + hand.boards.ai.middle.length + hand.boards.ai.bottom.length === 0
                ? "AIはFantasy Landを配置中 · カードはハンド終了まで非公開"
                : "手札と裏向き捨て札は非公開"
            }
          />

          {hand.to_act === "human" && hand.action_required === "normal" && hand.street ? (
            <PlacementWorkspace
              baseBoard={hand.boards.human}
              cards={hand.dealt_cards}
              street={hand.street}
              busy={submitting}
              onSubmit={submit}
            />
          ) : (
            <BoardView board={hand.boards.human} title="あなたの盤面" />
          )}

          {submitError ? <p className="inline-error" role="alert">{submitError}</p> : null}
          {hand.ai_pending || hand.to_act === "ai" ? (
            <div className="ai-overlay" role="status" aria-live="polite">
              <span className="ai-orbit" aria-hidden="true">
                <i />
              </span>
              <div>
                <strong>AI起動中</strong>
                <p>最適な配置を探索しています。画面は自動で更新されます。</p>
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </AppShell>
  );
}
