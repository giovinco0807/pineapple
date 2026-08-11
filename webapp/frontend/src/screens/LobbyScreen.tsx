import { useEffect, useState } from "react";
import { useNavigate } from "../router";
import {
  api,
  forgetMatch,
  getRecentMatchIds,
  getToken,
  rememberMatch,
  setToken
} from "../api";
import { handPath, errorMessage } from "../hooks";
import type { HandView, MatchView, MetaView } from "../types";
import { AppShell, ScreenHeading } from "../components/AppShell";

function positionLabel(match: MatchView): string {
  return match.first_hand_positions.human === "first" ? "初戦はあなたが先攻" : "初戦はAIが先攻";
}

interface ResumeApi {
  getHand(id: string): Promise<HandView>;
  startHand(matchId: string): Promise<HandView>;
}

export async function matchResumePath(
  match: MatchView,
  client: ResumeApi = api
): Promise<string> {
  if (match.status === "completed") return `/summary/${match.id}`;

  if (match.status === "awaiting_continue") {
    const latest = [...match.hands].sort((left, right) => right.index - left.index)[0];
    return latest ? `/result/${match.id}/${latest.id}` : `/summary/${match.id}`;
  }

  if (match.current_hand_id) {
    return handPath(await client.getHand(match.current_hand_id));
  }

  return handPath(await client.startHand(match.id));
}

export function LobbyScreen() {
  const navigate = useNavigate();
  const [tokenDraft, setTokenDraft] = useState(getToken());
  const [seed, setSeed] = useState("");
  const [resumeId, setResumeId] = useState("");
  const [recentIds, setRecentIds] = useState(getRecentMatchIds());
  const [recentMatches, setRecentMatches] = useState<MatchView[]>([]);
  const [meta, setMeta] = useState<MetaView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) return;
    let active = true;
    Promise.allSettled([api.getMeta(), ...recentIds.map((id) => api.getMatch(id))]).then((results) => {
      if (!active) return;
      const [metaResult, ...matches] = results;
      if (metaResult?.status === "fulfilled") setMeta(metaResult.value as MetaView);
      const loaded = matches
        .filter((item): item is PromiseFulfilledResult<MatchView> => item.status === "fulfilled")
        .map((item) => item.value);
      setRecentMatches(loaded);
    });
    return () => {
      active = false;
    };
  }, [recentIds]);

  const saveToken = () => {
    setToken(tokenDraft);
    setError(null);
    setRecentIds([...getRecentMatchIds()]);
    void api.getMeta().then(setMeta).catch(() => undefined);
  };

  const openMatch = async (match: MatchView) => {
    rememberMatch(match.id);
    setRecentIds([...getRecentMatchIds()]);
    navigate(await matchResumePath(match));
  };

  const create = async (forcedSeed?: number) => {
    if (!getToken()) {
      setError("先にアクセストークンを保存してください");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const numericSeed =
        forcedSeed ?? (seed.trim() && Number.isSafeInteger(Number(seed)) ? Number(seed) : undefined);
      const match = await api.createMatch(numericSeed == null ? {} : { seed: numericSeed });
      await openMatch(match);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const resume = async (id = resumeId) => {
    const clean = id.trim();
    if (!clean) return;
    setBusy(true);
    setError(null);
    try {
      await openMatch(await api.getMatch(clean));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const startDemo = async (fl = false) => {
    setToken("demo");
    setTokenDraft("demo");
    setBusy(true);
    setError(null);
    try {
      await openMatch(await api.createMatch(fl ? { seed: 1717 } : {}));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell>
      <div className="lobby-hero">
        <p className="kicker">HEADS-UP · NO JOKER</p>
        <h1>
          読み合いを、
          <br />
          <span>13枚に刻む。</span>
        </h1>
        <p>Regular OFC Pineapple。200点持ちで、学習済みAIと一対一。</p>
        <div className="hero-cards" aria-hidden="true">
          <span className="hero-card hero-card--one">
            A<small>♠</small>
          </span>
          <span className="hero-card hero-card--two">
            K<small>♥</small>
          </span>
          <span className="hero-card hero-card--three">
            Q<small>♦</small>
          </span>
        </div>
      </div>

      <section className="lobby-panel">
        <ScreenHeading eyebrow="START" title="対局をはじめる">
          <p>第1ハンドの先攻はサーバーがランダムに決めます。</p>
        </ScreenHeading>
        <label className="field">
          <span>アクセストークン</span>
          <span className="field__inline">
            <input
              type="password"
              value={tokenDraft}
              autoComplete="current-password"
              placeholder="Bearer token"
              onChange={(event) => setTokenDraft(event.target.value)}
            />
            <button type="button" className="button button--secondary" onClick={saveToken}>
              保存
            </button>
          </span>
          <small>この端末のブラウザ内にのみ保存されます。</small>
        </label>

        <div className="create-row">
          <label className="field field--seed">
            <span>シード（任意）</span>
            <input
              inputMode="numeric"
              value={seed}
              placeholder="ランダム"
              onChange={(event) => setSeed(event.target.value.replace(/[^\d-]/g, ""))}
            />
          </label>
          <button type="button" className="button button--primary button--large" disabled={busy} onClick={() => create()}>
            {busy ? "準備中…" : "新しいマッチ"}
          </button>
        </div>

        {error ? <p className="inline-error" role="alert">{error}</p> : null}

        <details className="demo-details">
          <summary>サーバーなしで画面を試す</summary>
          <div>
            <button type="button" className="button button--ghost" onClick={() => startDemo(false)}>
              通常対局デモ
            </button>
            <button type="button" className="button button--ghost" onClick={() => startDemo(true)}>
              14枚FLデモ
            </button>
          </div>
        </details>
      </section>

      <section className="lobby-section">
        <div className="section-label">
          <span>再開</span>
          <small>{recentMatches.length ? `${recentMatches.length}件の履歴` : "マッチIDから開く"}</small>
        </div>
        <div className="resume-row">
          <input
            value={resumeId}
            placeholder="マッチID"
            aria-label="再開するマッチID"
            onChange={(event) => setResumeId(event.target.value)}
          />
          <button type="button" className="button button--secondary" disabled={!resumeId.trim() || busy} onClick={() => resume()}>
            開く
          </button>
        </div>
        <div className="recent-list">
          {recentMatches.map((match) => (
            <article className="recent-match" key={match.id}>
              <button type="button" onClick={() => resume(match.id)}>
                <span>
                  <strong>
                    YOU {match.stacks.human} — {match.stacks.ai} AI
                  </strong>
                  <small>
                    {positionLabel(match)} · {match.hands.length}ハンド
                  </small>
                </span>
                <em>{match.status === "completed" ? "結果" : "再開"} →</em>
              </button>
              <button
                type="button"
                className="recent-match__forget"
                aria-label={`${match.id}を端末の履歴から削除`}
                onClick={() => {
                  forgetMatch(match.id);
                  setRecentIds([...getRecentMatchIds()]);
                  setRecentMatches((current) => current.filter((item) => item.id !== match.id));
                }}
              >
                ×
              </button>
            </article>
          ))}
        </div>
      </section>

      <footer className="lobby-meta">
        <span>REGULAR RULES</span>
        <span>200 POINTS</span>
        <span>{meta?.app_version ? `v${meta.app_version}` : "PWA"}</span>
      </footer>
    </AppShell>
  );
}
