import { useEffect, useState } from "react";
import { useNavigate, useParams } from "../router";
import { api } from "../api";
import { AppShell, ErrorState, LoadingState, ScoreRail } from "../components/AppShell";
import { PlacementWorkspace } from "../components/PlacementWorkspace";
import { errorMessage, handPath, useHand, useMatch } from "../hooks";
import type { PlacementPayload } from "../types";

export function FantasyLandScreen() {
  const { matchId = "", handId = "" } = useParams();
  const navigate = useNavigate();
  const { hand, setHand, loading, error, reload } = useHand(handId, true);
  const { match } = useMatch(matchId);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    if (hand && (hand.status === "complete" || hand.action_required !== "fl")) {
      navigate(handPath(hand), { replace: true });
    }
  }, [hand, navigate]);

  const submit = async (payload: PlacementPayload) => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const next = await api.submitFlPlacement(handId, payload);
      setHand(next);
      navigate(handPath(next), { replace: true });
    } catch (caught) {
      setSubmitError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AppShell matchId={matchId} compactHeader>
      {match ? <ScoreRail human={match.stacks.human} ai={match.stacks.ai} handIndex={hand?.index} /> : null}
      {loading && !hand ? <LoadingState label="Fantasy Landを準備しています" /> : null}
      {error && !hand ? <ErrorState message={error} onRetry={() => void reload()} /> : null}
      {hand ? (
        <>
          <section className="fl-hero">
            <span>FANTASY LAND</span>
            <h1>{hand.dealt_cards.length}枚を一括配置</h1>
            <p>13枚を3行に置き、残り{Math.max(0, hand.dealt_cards.length - 13)}枚を裏向きで捨てます。</p>
          </section>
          <div className="privacy-callout">
            <span aria-hidden="true">◈</span>
            <p>
              <strong>配置中の情報は非公開</strong>
              相手には完了したことだけが伝わり、盤面はハンド終了時に公開されます。
            </p>
          </div>
          <PlacementWorkspace
            baseBoard={hand.boards.human}
            cards={hand.dealt_cards}
            street="FL"
            fantasyLand
            busy={submitting}
            onSubmit={submit}
          />
          {submitError ? <p className="inline-error" role="alert">{submitError}</p> : null}
          {hand.ai_pending ? (
            <div className="ai-overlay" role="status" aria-live="polite">
              <span className="ai-orbit" aria-hidden="true">
                <i />
              </span>
              <div>
                <strong>AIのFLを計算中</strong>
                <p>厳密ソルバーの応答を待っています。</p>
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </AppShell>
  );
}
