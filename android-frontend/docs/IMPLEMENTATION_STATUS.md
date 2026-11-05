# SIP Implementation Status

Last reviewed: 2026-11-05 (Post-Recovery Verification & v0.3.0 Release).

This document reflects the verified operational state of SIP Guardian across production services, Android application releases, and edge inference components.

## Repository Architecture

This is a monorepo with three independently deployable components:
- `android-frontend/android/` — Android client (Java domain/data; Kotlin/Compose UI). Published release: `v0.3.0`.
- `database/backend/` — Spring Boot 3 / Java 17 REST + WebSocket API and Neon PostgreSQL persistence. **Canonical production backend** deployed on Render at `https://api.sip.bchbenjamin.in`.
- `landing-page/` — TanStack Start SSR frontend deployed on Vercel at `sip.bchbenjamin.in`.

The Raspberry Pi runs `node-backend/pi/` scripts for edge inference and autonomous training.
The Android app communicates only with `database/backend/` — never directly with Neon.

---

## Android (`v0.3.0`)

### Status: RELEASED & OPERATIONAL
- **Version:** `0.3.0` (versionCode 4).
- **GitHub Release:** [`v0.3.0`](https://github.com/bchbenjamin/SIP/releases/tag/v0.3.0) with signed production APK `sip-guardian-v0.3.0.apk`.
- **Signature:** Signed with `sip-beta.jks` using APK Signature Scheme v2.
- **R8 / ProGuard:** Explicit `keepclassmembers` rules guarantee zero field stripping/obfuscation on all network DTOs (`LoginResponse`, `IncidentResponse`, etc.) and domain entities.
- **Unit Testing:** 21/21 unit tests passing, including comprehensive coverage in `AuthRepositoryTest` for:
  - 200 OK login parsing and token persistence (`EncryptedSharedPreferences`)
  - Session refresh and JWT rotation
  - 401/403/500 error handling
  - Logout credential purging and subsequent re-login
- **Live Integration:** Verified live authentication against `https://api.sip.bchbenjamin.in/api/v1/auth/login`.

---

## Canonical Backend & Production Deployment

### Status: LIVE & VERIFIED
- **Production URL:** `https://api.sip.bchbenjamin.in/` (Render Web Service `srv-db3kh7gm7kps73f2f3bg`).
- **Live Deploy:** `dep-dbm2j2ivcj2c73814egg` built from commit `dab3457`.
- **Runtime:** Eclipse Temurin JDK 17 on Alpine Linux.
- **Database:** Neon PostgreSQL (HikariCP pool, max 5 connections).

### Live Endpoint Verification Matrix (2026-11-05)
| Endpoint | Method | Status | Notes |
| :--- | :--- | :---: | :--- |
| `/health` | GET | **PASS (200 OK)** | Returns `{"status":"UP"}` |
| `/actuator/health` | GET | **PASS (200 OK)** | Returns `{"service":"sip-backend","status":"UP"}` |
| `/api/v1/auth/login` | POST (Valid) | **PASS (200 OK)** | Returns HS512 JWT and user claims |
| `/api/v1/auth/login` | POST (Invalid) | **PASS (401)** | Rejects with `AUTH_INVALID_CREDENTIALS` |
| `/api/v1/nodes` | GET (JWT) | **PASS (200 OK)** | Returns edge nodes from PostgreSQL |
| `/api/v1/dashboard` | GET (JWT) | **PASS (200 OK)** | Returns aggregate system telemetry |
| `/api/v1/incidents` | GET (JWT) | **PASS (200 OK)** | Fixed in `dab3457` (bypasses unparameterized JPQL query) |
| `/api/v1/incidents` | POST (JWT) | **PASS (201 Created)** | Fixed in `dab3457` (maps entity to `IncidentDto`) |

---

## Raspberry Pi 5 Edge Compute Core

### Status: VERIFIED & OPERATIONAL
- **Hardware:** Raspberry Pi 5 Model B (Rev 1.0), 4.0 GiB RAM, Quad-core Cortex-A76 @ 2.4 GHz.
- **Active Systemd Units:**
  - `weapon-training.service` — YOLOv8 training pipeline
  - `sip-power-monitor.service` — Undervoltage and throttling monitor
  - `weapon-bot.service` — Interactive Telegram bot
- **Edge Inference Benchmark (YOLOv8n, 416x416 input):**
  - Mean latency: **378.63 ms** (**2.64 FPS**) under background training load
  - Min latency: **174.84 ms** (single-run best-case)
  - Max latency: **672.49 ms**
  - Median latency: **316.11 ms**
  - Thermal State: 71.9°C → 73.0°C with soft thermal limit/throttling (`0x50000`).
  - *Recommendation:* Training should not run concurrently with real-time detection in active security environments.
- **Model Checkpoints:**
  - Checkpointed 145-epoch model preserved at `/home/benjamin/runs/detect/train/weights/epoch145.pt` (0.816 mAP@50).

---

## CI/CD Status

| Workflow | Status | Branch | Commit |
| :--- | :---: | :--- | :--- |
| **Android CI** | **SUCCESS** | `main` | `dab3457` |
| **Backend CI** | **SUCCESS** | `main` | `dab3457` |

---

## Security & Credentials

- Database credentials, JWT secrets, and node secrets are injected strictly via environment variables.
- Keystores (`*.jks`) and `.env` files are ignored in `.gitignore`.
- SSH access to edge devices should use Ed25519 key authentication. Plaintext passwords must not be checked into documentation.
