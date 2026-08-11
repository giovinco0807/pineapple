import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { HandView, MatchView } from "./types";

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : "予期しないエラーが発生しました";
}

export function useMatch(matchId?: string) {
  const [match, setMatch] = useState<MatchView | null>(null);
  const [loading, setLoading] = useState(Boolean(matchId));
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!matchId) return;
    setLoading(true);
    setError(null);
    try {
      setMatch(await api.getMatch(matchId));
    } catch (caught) {
      setError(messageFrom(caught));
    } finally {
      setLoading(false);
    }
  }, [matchId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { match, setMatch, loading, error, reload };
}

export function useHand(handId?: string, polling = false) {
  const [hand, setHand] = useState<HandView | null>(null);
  const [loading, setLoading] = useState(Boolean(handId));
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!handId) return null;
    setLoading(true);
    setError(null);
    try {
      const next = await api.getHand(handId);
      setHand(next);
      return next;
    } catch (caught) {
      setError(messageFrom(caught));
      return null;
    } finally {
      setLoading(false);
    }
  }, [handId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!polling || !handId || !(hand?.ai_pending || hand?.to_act === "ai") || hand.status === "complete") return;
    let stopped = false;
    let timeout: number | undefined;
    const poll = async () => {
      try {
        const next = await api.getHand(handId);
        if (stopped) return;
        setHand(next);
        setError(null);
        if ((next.ai_pending || next.to_act === "ai") && next.status !== "complete") {
          timeout = window.setTimeout(poll, 1000);
        }
      } catch (caught) {
        if (stopped) return;
        setError(messageFrom(caught));
        timeout = window.setTimeout(poll, 1800);
      }
    };
    timeout = window.setTimeout(poll, 1000);
    return () => {
      stopped = true;
      if (timeout) window.clearTimeout(timeout);
    };
  }, [hand?.ai_pending, hand?.status, hand?.to_act, handId, polling]);

  return { hand, setHand, loading, error, reload };
}

export function handPath(hand: HandView): string {
  if (hand.status === "complete") return `/result/${hand.match_id}/${hand.id}`;
  if (hand.action_required === "fl" && hand.to_act === "human") return `/fl/${hand.match_id}/${hand.id}`;
  return `/play/${hand.match_id}/${hand.id}`;
}

export function errorMessage(error: unknown): string {
  return messageFrom(error);
}
