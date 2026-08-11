import { useEffect, useMemo, useState } from "react";
import { useParams } from "../router";
import { api } from "../api";
import { AppShell, ErrorState, LoadingState, ScreenHeading } from "../components/AppShell";
import { errorMessage, useMatch } from "../hooks";

export function ExportScreen() {
  const { matchId = "" } = useParams();
  const { match } = useMatch(matchId);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [preview, setPreview] = useState<string[]>([]);
  const [lineCount, setLineCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const nextBlob = await api.exportMatch(matchId);
      const text = await nextBlob.text();
      const lines = text.split(/\r?\n/).filter(Boolean);
      setBlob(nextBlob);
      setLineCount(lines.length);
      setPreview(lines.slice(0, 4));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [matchId]);

  const sizeLabel = useMemo(() => {
    if (!blob) return "—";
    if (blob.size < 1024) return `${blob.size} B`;
    return `${(blob.size / 1024).toFixed(1)} KB`;
  }, [blob]);

  const download = () => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `ofc-match-${matchId}.jsonl`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <AppShell matchId={matchId}>
      <ScreenHeading eyebrow="EXPORT" title="対局記録を持ち出す">
        <p>1行1決定とハンドサマリーを含むJSONLです。学習・分析・完全再生に使えます。</p>
      </ScreenHeading>
      {loading ? <LoadingState label="JSONLを準備しています" /> : null}
      {error ? <ErrorState message={error} onRetry={() => void load()} /> : null}
      {!loading && blob ? (
        <>
          <section className="export-card">
            <span className="export-card__icon" aria-hidden="true">
              {"{ }"}
            </span>
            <div>
              <small>APPLICATION / X-NDJSON</small>
              <strong>ofc-match-{matchId.slice(0, 12)}.jsonl</strong>
              <p>
                {lineCount}行 · {sizeLabel} · {match?.hands.length ?? "—"}ハンド
              </p>
            </div>
            <span className="verified-chip">READY</span>
          </section>
          <button type="button" className="button button--primary button--large download-button" onClick={download}>
            JSONLをダウンロード
          </button>

          <section className="export-includes">
            <div className="section-label">
              <span>含まれる記録</span>
              <small>サーバー保存データ</small>
            </div>
            <ul>
              <li>
                <span>01</span>
                全ストリートの配札・配置・捨て札
              </li>
              <li>
                <span>02</span>
                AI evaluator・重みSHA・思考時間・上位3手
              </li>
              <li>
                <span>03</span>
                公式採点内訳・点数遷移・ハンドサマリー
              </li>
            </ul>
          </section>

          <details className="json-preview">
            <summary>先頭4行を確認</summary>
            <pre>
              {preview.map((line, index) => (
                <code key={index}>{line}</code>
              ))}
            </pre>
          </details>
          <p className="privacy-footnote">エクスポートには対局終了後の完全な記録が含まれます。共有時は取り扱いに注意してください。</p>
        </>
      ) : null}
    </AppShell>
  );
}
