import { HashRouter, Navigate, RouteParamsProvider, useLocation } from "./router";
import type { ReactNode } from "react";
import { ExportScreen } from "./screens/ExportScreen";
import { FantasyLandScreen } from "./screens/FantasyLandScreen";
import { GameScreen } from "./screens/GameScreen";
import { HandResultScreen } from "./screens/HandResultScreen";
import { HistoryScreen } from "./screens/HistoryScreen";
import { LobbyScreen } from "./screens/LobbyScreen";
import { MatchSummaryScreen } from "./screens/MatchSummaryScreen";

type RouteMatch = { element: ReactNode; params: Record<string, string> };

function matchRoute(pathname: string): RouteMatch | null {
  const routes: Array<{ pattern: RegExp; keys: string[]; element: ReactNode }> = [
    { pattern: /^\/$/, keys: [], element: <LobbyScreen /> },
    { pattern: /^\/play\/([^/]+)\/([^/]+)$/, keys: ["matchId", "handId"], element: <GameScreen /> },
    { pattern: /^\/fl\/([^/]+)\/([^/]+)$/, keys: ["matchId", "handId"], element: <FantasyLandScreen /> },
    { pattern: /^\/result\/([^/]+)\/([^/]+)$/, keys: ["matchId", "handId"], element: <HandResultScreen /> },
    { pattern: /^\/summary\/([^/]+)$/, keys: ["matchId"], element: <MatchSummaryScreen /> },
    { pattern: /^\/history\/([^/]+)$/, keys: ["matchId"], element: <HistoryScreen /> },
    { pattern: /^\/export\/([^/]+)$/, keys: ["matchId"], element: <ExportScreen /> }
  ];
  for (const route of routes) {
    const match = pathname.match(route.pattern);
    if (!match) continue;
    const params = Object.fromEntries(route.keys.map((key, index) => [key, decodeURIComponent(match[index + 1])]));
    return { element: route.element, params };
  }
  return null;
}

function RoutedApp() {
  const { pathname } = useLocation();
  const matched = matchRoute(pathname);
  if (!matched) return <Navigate to="/" replace />;
  return <RouteParamsProvider params={matched.params}>{matched.element}</RouteParamsProvider>;
}

export default function App() {
  return (
    <HashRouter>
      <RoutedApp />
    </HashRouter>
  );
}
