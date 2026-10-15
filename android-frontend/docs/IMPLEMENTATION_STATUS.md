# SIP Implementation Status

Last reviewed: 2026-10-15.

This status reflects the current repository source and the latest GitHub Actions checks. A passing CI build is not proof of a live Neon deployment, physical-device connectivity, or Raspberry Pi integration.

## Repository architecture

This is a monorepo with three independently deployable components:
- `android-frontend/android/` — Android client (Java domain/data; Kotlin/Compose UI).
- `database/backend/` — Spring Boot 3 / Java 17 REST + WebSocket API and Neon PostgreSQL persistence. **Canonical production backend** deployed on Render at `api.sip.bchbenjamin.in`.
- `landing-page/` — TanStack Start SSR frontend deployed on Vercel at `sip.bchbenjamin.in`.

The Pi runs `node-backend/pi/` scripts for edge inference and autonomous training.
The Android app communicates only with `database/backend/` — never directly with Neon.

**Note:** `android-frontend/backend/` is a deprecated duplicate, not used in CI/CD.

## Android

Implemented in source:
- Compose shell, login, dashboard, incident feed/detail and operator verification flow.
- Domain models, incident-state validation, Hilt dependency injection and Room cache.
- Retrofit/OkHttp REST integration, encrypted token storage and refresh authenticator.
- Authenticated WebSocket client, reconnect backoff, parser and foreground service.
- Full incident-update events refresh Room; incident notifications deep-link to matching detail screen.
- DTO/mapping compatibility for incident payloads and unit tests for state/policy/parser behavior.
- Debug APK available in `releases/sip-guardian-debug-latest.apk` (16MB).

Outstanding:
- Release APK (requires correct keystore password for `sip-beta.jks`).
- FCM delivery when app is killed/backgrounded and WebSocket service is unavailable.
- Full Android/backend end-to-end tests against live backend.

## Backend

Implemented in source:
- REST application, JPA entities/repositories, Flyway migrations and PostgreSQL configuration.
- JWT login, refresh-token persistence/rotation, logout/revocation and optional environment-based bootstrap administrator.
- Incident feed/detail and operator verification, node reads, dashboard, autopilot policy endpoints, audit records and WebSocket broadcast.
- Global API error format, request IDs, health endpoint and a separate backend CI workflow.
- Pi incident ingestion endpoint (REST fallback when MQTT unavailable).
- Incident state machine with 11 states: DETECTED → EVIDENCE_CAPTURED → AUTONOMOUS_EVALUATION → AUTO_HANDLED → PENDING_VERIFICATION → VERIFIED/REJECTED/ESCALATED → DETERRENCE_ACTIVE → DETERRENCE_COMPLETED → RESOLVED.
- Separate incident service package: `IncidentService`, `IncidentMapper`, `IncidentStateMachine`.

Smoke test results (2026-10-15):
| Endpoint | Status |
|---|---|
| GET /actuator/health | PASS |
| POST /api/v1/auth/login | PASS |
| GET /api/v1/nodes | PASS |
| GET /api/v1/dashboard | PASS |
| GET /api/v1/incidents | FAIL (500 — under investigation) |

Outstanding:
- Debug /api/v1/incidents 500 error (incident service package missing in source until 3c9de72).
- Full WebSocket authorization/channel subscription semantics and REST resynchronization contract tests.
- MQTT ingestion/command bridge, node-to-backend authentication, heartbeat ingestion and idempotent Pi incident ingestion.
- Evidence object storage, signed URLs, upload retries and retention enforcement.
- User administration, password reset/recovery, rate limiting and deployment-level HTTPS/secret management.
- Integration tests against PostgreSQL and end-to-end Android/backend tests.

## Current audit fixes

- Dashboard recent incidents are queried through a SQL-limited top-10 repository method rather than an unbounded query. The active-threat total includes every non-terminal incident stage, including autonomous evaluation and active deterrence.
- Incident-update WebSocket notification is scheduled after transaction commit.
- Autopilot policy JSON exposes `lastSyncTimestamp`, matching the Android DTO.
- Room now uses an explicit v1→v2 migration instead of destructive fallback, and incident response events survive the local-cache round trip. Offline incident cache queries preserve filters and pagination rather than showing unrelated cached incidents.
- IncidentState enum aligned with state machine (11 states, matching DB constraint V5__align_incident_lifecycle.sql).

## Raspberry Pi

Implemented in source (`node-backend/pi/`):
- `edge_server.py` — WebSocket inference server (YOLOv8 threat detection).
- `train_autonomous.py` — Autonomous training with dataset manifest verification and checkpoint compatibility checks.
- `download_and_merge.py` — Downloads Zenodo NESTLER, Kaggle Pet Detection, TigDog datasets and merges into unified YOLO format.
- `dataset_registry.py` — Persistent JSON registry with atomic writes, tracks 7 datasets (2 TRAINING_ELIGIBLE, 5 pending download).
- `dataset_audit.py` — Forensic audit: image decode, YOLO label validation, class ID range, duplicate detection. Audit passed: 17924 samples, 0 malformed, 0 invalid class.
- `power_monitor.py` — Independent systemd service, polls `vcgencmd get_throttled` every 60s, sends ONE Telegram alert after >30 min undervoltage, one recovery message.
- `sip-power-monitor.service` — systemd unit for power monitor (Nice=5, Restart=always).
- `telegram_notify.py` — Progress notifications with IPv4/IPv6 display.
- `register_datasets.py` — Pre-populates registry with all known datasets.

Dataset classes: `["gun", "knife", "dangerous_animal"]` (weapon detector separate from planned animal specialist).

Outstanding:
- Download and validate Zenodo NESTLER, Kaggle Pet Detection, and TigDog datasets.
- Build and train animal specialist model (separate from weapon detector).
- Fix release APK signing (keystore password for `sip-beta.jks`).

## Environment and secrets

- Backend requires `DATABASE_URL` (a PostgreSQL JDBC connection string) and a strong `JWT_SECRET`.
- Optional first-admin bootstrap requires both `SIP_BOOTSTRAP_ADMIN_USERNAME` and `SIP_BOOTSTRAP_ADMIN_PASSWORD`.
- A Neon management API key is not a database connection string.
- Pi credentials: `benjamin` / `13122006` (reachable via WiFi `192.168.0.125` or USB gadget `10.42.0.1`).
- Backend admin: `bchbenjamin` / `S1pGuardian2026!` (Neon seeded).
- Never put database credentials, JWT secrets, node credentials or SSH credentials in Android resources, BuildConfig, the APK or version control.

## Verification

- Android CI builds and tests the debug app.
- Backend CI builds/tests the Spring Boot service.
- Pi scripts: forensic audit PASSED, power monitor ACTIVE, dataset registry POPULATED.
- These checks do not prove live Neon connectivity, physical-device reachability, Pi integration or production readiness.
