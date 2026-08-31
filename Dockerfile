FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY t0_tutor_app.py .
COPY training_presets.json .
COPY tutor_static/ tutor_static/

COPY backend/__init__.py backend/ai_player.py backend/
COPY ai/__init__.py ai/
COPY ai/engine/ ai/engine/
COPY ai/mcts/ ai/mcts/
COPY ai/config/ ai/config/
COPY ai/models/__init__.py ai/models/networks.py ai/models/
COPY ai/models/expectimax_bc_v3/bc_policy_best.pt ai/models/expectimax_bc_v3/bc_policy_best.pt
COPY ai/models/value_v3/value_best.pt ai/models/value_v3/value_best.pt
COPY ai/models/value_v3/norm_stats.json ai/models/value_v3/norm_stats.json
COPY ai/data/tutor_route10_20260522/tutor_route10_review.json ai/data/tutor_route10_20260522/tutor_route10_review.json

EXPOSE 8080

CMD ["python", "t0_tutor_app.py"]
