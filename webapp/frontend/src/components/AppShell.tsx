import type { ReactNode } from "react";
import { NavLink, useLocation } from "../router";

interface AppShellProps {
  children: ReactNode;
  matchId?: string;
  compactHeader?: boolean;
}

export function AppShell({ children, matchId, compactHeader = false }: AppShellProps) {
  const location = useLocation();
  const historyTarget = matchId ? `/history/${matchId}` : "/";
  const exportTarget = matchId ? `/export/${matchId}` : "/";

  return (
    <div className="app-frame">
      <div className="portrait-lock" role="status">
        <span aria-hidden="true">↻</span>
        <strong>端末を縦向きにしてください</strong>
      </div>
      <header className={`app-header ${compactHeader ? "app-header--compact" : ""}`}>
        <NavLink to="/" className="brand" aria-label="OFC Pineapple ホーム">
          <span className="brand__mark" aria-hidden="true">
            ♠
          </span>
          <span>
            <strong>OFC</strong>
            <small>PINEAPPLE</small>
          </span>
        </NavLink>
        {matchId ? <span className="match-chip">#{matchId.slice(0, 8)}</span> : <span className="rule-chip">REGULAR</span>}
      </header>
      <main className="app-main">{children}</main>
      <nav className="bottom-nav" aria-label="メインメニュー">
        <NavLink to="/" className={location.pathname === "/" ? "active" : ""}>
          <span aria-hidden="true">⌂</span>
          <small>ホーム</small>
        </NavLink>
        <NavLink to={historyTarget} aria-disabled={!matchId} className={!matchId ? "disabled" : ""}>
          <span aria-hidden="true">↺</span>
          <small>履歴</small>
        </NavLink>
        <NavLink to={exportTarget} aria-disabled={!matchId} className={!matchId ? "disabled" : ""}>
          <span aria-hidden="true">⇩</span>
          <small>出力</small>
        </NavLink>
      </nav>
    </div>
  );
}

export function ScreenHeading({
  eyebrow,
  title,
  children
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="screen-heading">
      {eyebrow ? <p>{eyebrow}</p> : null}
      <h1>{title}</h1>
      {children}
    </div>
  );
}

export function ScoreRail({
  human,
  ai,
  handIndex
}: {
  human: number;
  ai: number;
  handIndex?: number;
}) {
  const humanPct = Math.max(0, Math.min(100, (human / Math.max(human + ai, 1)) * 100));
  return (
    <section className="score-rail" aria-label={`スコア あなた${human}点、AI${ai}点`}>
      <div className="score-rail__numbers">
        <span>
          <small>YOU</small>
          <strong>{human}</strong>
        </span>
        {handIndex ? <em>HAND {handIndex}</em> : <em>MATCH</em>}
        <span>
          <small>AI</small>
          <strong>{ai}</strong>
        </span>
      </div>
      <div className="score-rail__track">
        <span style={{ width: `${humanPct}%` }} />
      </div>
    </section>
  );
}

export function LoadingState({ label = "読み込み中" }: { label?: string }) {
  return (
    <div className="state-panel" role="status">
      <span className="spinner" aria-hidden="true" />
      <strong>{label}</strong>
      <p>サーバーの応答を待っています</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-panel state-panel--error" role="alert">
      <span aria-hidden="true">!</span>
      <strong>うまく接続できませんでした</strong>
      <p>{message}</p>
      {onRetry ? (
        <button type="button" className="button button--secondary" onClick={onRetry}>
          もう一度試す
        </button>
      ) : null}
    </div>
  );
}
