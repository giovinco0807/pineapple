# T0 Tutor Monetization Audit

Date: 2026-05-23

## Saved Tutor Data

- Review Markdown: `ai/data/tutor_route10_20260522/tutor_route10_review.md`
- Merged JSONL: `ai/data/tutor_route10_20260522/tutor_route10_merged.jsonl`
- UI-ready JSON: `ai/data/tutor_route10_20260522/tutor_route10_review.json`
- Source run: `tutor-route10-p80-s300-b14-c4-tr96-20260522`
- Contents: 10 hands, 20 BB/BTN records, high-precision route traces through T1-T4.

## UI Inventory

There are two UI surfaces in this repository.

1. `frontend/` + `backend/main.py`
   - This is the deployed Docker/Render path.
   - It is a real-time OFC game room UI with WebSocket play and AI-room support.
   - It has no user accounts, subscription gating, or Stripe checkout.

2. `t0_tutor_app.py` + `tutor_static/index.html`
   - This is the closer monetization candidate.
   - It already has plan names, usage limits, Stripe Checkout, Firebase Admin verification, and a T0 tutor screen.
   - It is not currently wired into the Docker/Render deployment path.

## Monetization Blockers

1. Frontend auth is mocked.
   - `tutor_static/index.html` signs in as `demo@ofc-tutor.dev`, sets `mock-token-demo`, and marks the user as `premium`.
   - The backend requires a real Firebase ID token, so production API calls will fail unless real Firebase web auth is added.

2. Training mode does not use backend presets.
   - The backend has `/api/training_puzzle` and `/api/evaluate_training`.
   - The frontend currently deals deterministic local cards and sends `currentPuzzleId`, which stays null.
   - That means server-side usage limits and preset labels are not aligned with the UI.

3. Stripe is only a partial implementation.
   - Checkout exists, but only one price id is supported.
   - There is no billing portal.
   - Webhooks only handle `checkout.session.completed`; cancellation and subscription updates are not handled.

4. The deploy path does not run the tutor app.
   - `Dockerfile` starts `backend/main.py`, not `t0_tutor_app.py`.
   - `requirements.txt` lacks tutor dependencies such as `firebase-admin` and `stripe`.

5. Some UI text is mojibake.
   - The React game UI and parts of `tutor_static/index.html` include corrupted Japanese strings.
   - This should be fixed before public launch.

## Recommended Monetizable MVP

Use `t0_tutor_app.py` as the product base, not the real-time game UI.

Free tier:
- 3-10 free T0 puzzles per day.
- Show T0 only.
- Show top 3 placements and basic FL/AA metrics.

Paid tier:
- Full custom T0 evaluation.
- High-precision route review hands.
- T1-T4 correct-route playback from the generated tutor data.
- More daily volume.

Implementation order:

1. Add real Firebase web login to `tutor_static/index.html`.
2. Make training mode fetch `/api/training_puzzle` instead of local deterministic cards.
3. Add Stripe Starter/Premium price ids and plan metadata.
4. Add Stripe billing portal and subscription update/delete webhook handling.
5. Add a paid review API/page backed by `tutor_route10_review.json`.
6. Point a deployment target at the tutor app, or merge tutor routes into `backend/main.py`.
7. Fix mojibake and verify the UI in-browser.

