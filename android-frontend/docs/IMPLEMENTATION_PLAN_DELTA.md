# Implementation Plan Delta — Monorepo + Live Backend Audit

**Applies to:** `bchbenjamin/SIP-Android`  
**Reviewed:** 2026-10-01  
**Supersedes conflicting repository/status statements in:** `CONTEXT/SIP_backend_neon_implementation_plan.md`

The original plan remains the project requirements/reference. This delta records implementation decisions that have since been made and the current gaps. Do not follow the old proposal to create a separate `SIP-Backend` repository: the implementation is a monorepo.

## Canonical architecture

```text
Raspberry Pi edge node ── future MQTT/HTTPS ──► Spring Boot backend ── JDBC ──► Neon PostgreSQL
                                                    ▲
                                                    │ HTTPS / WSS
                                                    │
                                              Android app
                                                    │
                                                Room cache
```

- `/android` is the Android client.
- `/backend` is the Spring Boot service.
- `/docs` contains maintained status and implementation notes.
- Neon credentials belong only in backend runtime environment configuration.
- Android must not connect directly to Neon, MQTT, or Raspberry Pi.
- Room is a local cache, not the system of record.
- Raspberry Pi local inference and autonomous response must remain available during network outages.

## Implemented in repository source

- Spring Boot 3 / Java 17 REST service, JPA entities/repositories, Flyway migrations and PostgreSQL configuration.
- JWT login and refresh-token rotation/revocation, optional environment-configured bootstrap administrator.
- Incident feed/detail/verification, node reads, dashboard summary, autopilot-policy API and audit records.
- Authenticated WebSocket handshake and event broadcasting.
- Android Retrofit API, encrypted token store, refresh authenticator, Room cache, Compose login/dashboard/incident screens.
- Separate Android and backend CI workflows.

## Important limits (not implied complete)

- No live Neon connection or clean Neon migration has been verified by CI alone.
- No Pi MQTT publisher/subscriber, node authentication, heartbeat ingestion or incident-ingestion API is complete.
- Evidence media routes explicitly remain unavailable until an object-storage provider is implemented.
- No FCM delivery, production Room migrations, full end-to-end Android/backend tests, or production deployment hardening.
- UI for node map, autopilot policy, event history, settings and evidence timeline remains outstanding.

## Audit fixes recorded by this delta

- Dashboard's recent-incident query must be SQL-limited to ten rows; never fetch the entire incident table for a dashboard widget.
- Incident-update WebSocket events must be emitted only after the database transaction commits.
- Autopilot policy JSON must use the Android contract field `lastSyncTimestamp`.\n- Room schema changes must use explicit migrations (never destructive fallback), and cached incident response events must be preserved.
- Full `INCIDENT_UPDATED` WebSocket payloads must update Room, and new-incident notifications must deep-link to the incident detail route.\n- Dashboard active-threat counts must include all non-terminal states; offline incident queries must apply the same filters and pagination as the API.

## Required workflow

1. Read `CONTEXT/CONTEXT.md` and the original plan.
2. Implement backend contract and tests first, then Android integration.
3. Run backend tests/build and Android tests/build.
4. Verify clean migrations against a disposable PostgreSQL/Neon development database when credentials are available.
5. Do not begin Pi integration until Android ↔ backend REST/auth/real-time synchronization is verified.
6. Update `README.md`, `docs/IMPLEMENTATION_STATUS.md`, API docs and this delta whenever implementation status changes.
7. Never commit secrets; keep examples blank and rotate any exposed credentials.
