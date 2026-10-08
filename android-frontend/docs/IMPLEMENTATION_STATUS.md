# SIP Implementation Status

Last reviewed: 2026-10-01.

This status reflects the current repository source and the latest GitHub Actions checks. A passing CI build is not proof of a live Neon deployment, physical-device connectivity, or Raspberry Pi integration.

## Repository architecture

This is a monorepo:
- `android/` — Android client (Java domain/data/business logic; Kotlin/Compose UI).
- `backend/` — Spring Boot 3 / Java 17 REST + WebSocket API and PostgreSQL persistence.
- `CONTEXT/` — project context and original implementation plan.
- `docs/IMPLEMENTATION_PLAN_DELTA.md` — authoritative delta for monorepo decisions and remaining scope.

Neon/PostgreSQL is the canonical structured-data store. Android talks only to the backend; Room is a local cache. Raspberry Pi local inference and autonomous actions must continue if the backend is unreachable.

## Android

Implemented in source:
- Compose shell, login, dashboard, incident feed/detail and operator verification flow.
- Domain models, incident-state validation, Hilt dependency injection and Room cache.
- Retrofit/OkHttp REST integration, encrypted token storage and refresh authenticator.
- Authenticated WebSocket client, reconnect backoff, parser and foreground service. Full incident-update events now refresh Room; incident notifications deep-link to the matching detail screen.
- DTO/mapping compatibility for incident payloads and unit tests for state/policy/parser behavior.

Outstanding:
- Node map, autopilot policy UI, event history, settings and evidence/media timeline.
- FCM delivery when the app is killed/backgrounded and WebSocket service is unavailable.
- Full Android/backend contract or end-to-end tests.
- Physical-device verification against a reachable HTTPS backend.
- Full Android/backend contract tests against a live or disposable backend.

## Backend

Implemented in source:
- REST application, JPA entities/repositories, Flyway migrations and PostgreSQL configuration.
- JWT login, refresh-token persistence/rotation, logout/revocation and optional environment-based bootstrap administrator.
- Incident feed/detail and operator verification, node reads, dashboard, autopilot policy endpoints, audit records and WebSocket broadcast.
- Global API error format, request IDs, health endpoint and a separate backend CI workflow.

Outstanding:
- Live Neon provisioning/connectivity and clean-database migration test against PostgreSQL.
- Full WebSocket authorization/channel subscription semantics and REST resynchronization contract tests.
- MQTT ingestion/command bridge, node-to-backend authentication, heartbeat ingestion and idempotent Pi incident ingestion.
- Evidence object storage, signed URLs, upload retries and retention enforcement.
- User administration, password reset/recovery, rate limiting and deployment-level HTTPS/secret management.
- Integration tests against PostgreSQL and end-to-end Android/backend tests.

## Current audit fixes

- Dashboard recent incidents are queried through a SQL-limited top-10 repository method rather than an unbounded query. The active-threat total includes every non-terminal incident stage, including autonomous evaluation and active deterrence.
- Incident-update WebSocket notification is scheduled after transaction commit.
- Autopilot policy JSON exposes `lastSyncTimestamp`, matching the Android DTO.\n- Room now uses an explicit v1→v2 migration instead of destructive fallback, and incident response events survive the local-cache round trip. Offline incident cache queries preserve filters and pagination rather than showing unrelated cached incidents.

## Environment and secrets

- Backend requires `DATABASE_URL` (a PostgreSQL JDBC connection string) and a strong `JWT_SECRET`.
- Optional first-admin bootstrap requires both `SIP_BOOTSTRAP_ADMIN_USERNAME` and `SIP_BOOTSTRAP_ADMIN_PASSWORD`.
- A Neon management API key is not a database connection string.
- Never put database credentials, JWT secrets, node credentials or SSH credentials in Android resources, BuildConfig, the APK or version control.

## Verification

- Android CI builds and tests the debug app.
- Backend CI builds/tests the Spring Boot service.
- These checks do not prove live Neon connectivity, physical-device reachability, Pi integration or production readiness.
