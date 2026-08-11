# OFC Pineapple frontend

縦持ちスマートフォン向けの React + TypeScript + Vite + Tailwind PWA です。

```powershell
npm install
npm run dev
npm test
npm run build
```

FastAPI と同一オリジンで配信する場合、追加設定は不要です。開発中に API
を別ホストへ向ける場合は `.env.example` を `.env.local` にコピーして
`VITE_API_ROOT` を指定します。

ロビーのアクセストークンは `localStorage` に保存され、全 API
リクエストの `Authorization: Bearer ...` に使われます。サーバーなしで
画面を確認するときはロビーの「通常対局デモ」または「14枚FLデモ」を
選べます。
