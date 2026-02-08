# Stage 1: Build frontend
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Build Rust solver
FROM rust:1.75-slim AS rust-build
WORKDIR /app/ai/rust_solver
COPY ai/rust_solver/ .
RUN cargo build --release

# Stage 3: Production
FROM python:3.11-slim
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ backend/
COPY ai/engine/ ai/engine/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Copy built Rust solver
COPY --from=rust-build /app/ai/rust_solver/target/release/fl_solver ai/rust_solver/target/release/fl_solver

# Create data directory
RUN mkdir -p data

EXPOSE 8080

CMD ["python", "backend/main.py"]
