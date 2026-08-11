# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /build/webapp/frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY webapp/frontend/ ./
RUN npm run build

FROM rust:1.93-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY rust/regular_fl_solver/ rust/regular_fl_solver/
COPY rust/ofc_stage3_feature_encoder/ rust/ofc_stage3_feature_encoder/
COPY rust/hu_m3_engine/Cargo.toml rust/hu_m3_engine/Cargo.toml
COPY rust/hu_m3_engine/src/ rust/hu_m3_engine/src/
COPY webapp/rust/decision_inspector/ webapp/rust/decision_inspector/
RUN cargo build --locked --release --package ofc_hu_m3_engine --lib
RUN cargo build --locked --release --package regular_fl_solver --bin regular_fl_solver --lib
RUN cargo build \
    --locked \
    --manifest-path webapp/rust/decision_inspector/Cargo.toml \
    --release

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/backend:/app \
    LD_LIBRARY_PATH=/app/native \
    OFC_ASSEMBLY_PATH=/app/assembly.json \
    OFC_ASSEMBLY_ROOT=/app \
    OFC_ENGINE_LIBRARY=/app/native/libofc_hu_m3_engine.so \
    OFC_FEATURE_ENCODER_LIBRARY=/app/native/libofc_stage3_feature_encoder.so \
    OFC_FL_SOLVER_PATH=/app/native/regular_fl_solver \
    OFC_DECISION_INSPECTOR_PATH=/app/native/ofc_webapp_decision_inspector \
    OFC_STATIC_DIR=/app/static \
    OFC_DB_PATH=/tmp/ofc-webapp.sqlite3 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.txt

COPY src/ofc_regular/ /app/src/ofc_regular/
COPY webapp/backend/ /app/backend/
COPY webapp/assembly.json /app/assembly.json
COPY webapp/deploy/ /app/deploy/
COPY --from=frontend-builder /build/webapp/frontend/dist/ /app/static/

COPY --from=rust-builder /build/target/release/libofc_hu_m3_engine.so /app/native/libofc_hu_m3_engine.so
COPY --from=rust-builder /build/target/release/libofc_stage3_feature_encoder.so /app/native/libofc_stage3_feature_encoder.so
COPY --from=rust-builder /build/target/release/regular_fl_solver /app/native/regular_fl_solver
COPY --from=rust-builder /build/webapp/rust/decision_inspector/target/release/ofc_webapp_decision_inspector /app/native/ofc_webapp_decision_inspector

COPY rust/hu_m3_engine/tests/fixtures/t4_model_v5.bin /app/models/learned/t4_model_v5.bin
COPY rust/hu_m3_engine/tests/fixtures/t3_model_v2.bin /app/models/learned/t3_model_v2.bin
COPY rust/hu_m3_engine/tests/fixtures/t3first_model_v1.bin /app/models/learned/t3first_model_v1.bin
COPY rust/hu_m3_engine/tests/fixtures/t2_model_v1.bin /app/models/learned/t2_model_v1.bin

COPY models/opening_stage7_torch_wide.pt /app/models/opening_stage7_torch_wide.pt
COPY models/turn1_stage6_torch_wide.pt /app/models/turn1_stage6_torch_wide.pt
COPY models/turn2_stage8.pkl /app/models/turn2_stage8.pkl
COPY models/turn3_stage6.pkl /app/models/turn3_stage6.pkl
COPY models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl /app/models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl
COPY models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl /app/models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl
COPY models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl /app/models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl
COPY models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl /app/models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl
COPY models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt /app/models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt
COPY models/hu_turn3_stage7_reference_override_cached_rank_wide.pt /app/models/hu_turn3_stage7_reference_override_cached_rank_wide.pt
COPY models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt /app/models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt

RUN chmod 0555 \
      /app/deploy/entrypoint.sh \
      /app/native/regular_fl_solver \
      /app/native/ofc_webapp_decision_inspector \
    && python /app/deploy/verify_assembly.py \
      --assembly /app/assembly.json \
      --root /app \
      --seal-build-artifacts \
    && chmod 0444 /app/assembly.json \
    && python /app/deploy/verify_assembly.py \
      --assembly /app/assembly.json \
      --root /app \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin ofcweb

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["serve"]
