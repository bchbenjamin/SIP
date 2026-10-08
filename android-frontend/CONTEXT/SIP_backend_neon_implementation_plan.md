# SIP Guardian / AI-Powered Street Safety Device Network
## Backend + Neon PostgreSQL Implementation Plan

**Project IDs:** ASIP_127 / 24UTAI13  
**Android repository:** `bchbenjamin/SIP-Android`  
**Proposed backend repository:** `bchbenjamin/SIP-Backend` (name can be changed)  
**Database:** Neon PostgreSQL  
**Status:** Planning / implementation-ready

---

## 0. The Architectural Decision

The new design should make the **backend the single system-of-record and control plane**.

The Android app must **never connect directly to PostgreSQL/Neon, MQTT, or a Raspberry Pi**.
The Raspberry Pi must also stop treating the Android app as a destination for operational data.

The intended topology is:

```text
                         ┌──────────────────────────┐
                         │       Neon PostgreSQL    │
                         │   SYSTEM OF RECORD      │
                         └────────────▲─────────────┘
                                      │ JDBC / JPA
                                      │
                         ┌────────────┴─────────────┐
                         │      SIP BACKEND         │
                         │      Spring Boot         │
                         │                          │
                         │ REST API                 │
                         │ WebSocket                │
                         │ Authentication           │
                         │ MQTT bridge              │
                         │ Incident services        │
                         │ Node services            │
                         │ Evidence services        │
                         └───────▲──────────┬───────┘
                                 │          │
                         HTTPS/WS │          │ MQTT/HTTPS
                                 │          │
                 ┌───────────────┘          └───────────────┐
                 │                                           │
        ┌────────┴────────┐                         ┌────────┴─────────┐
        │ Android App    │                         │ Raspberry Pi(s)  │
        │ SIP Guardian   │                         │ Edge AI Nodes    │
        └────────────────┘                         └──────────────────┘
```

### Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| Raspberry Pi | Sensor acquisition, AI inference, local autonomy, deterrence | Android UI, canonical cloud data store |
| SIP Backend | API, auth, canonical incident state, node state, audit trail, WebSocket events, MQTT bridge | AI inference |
| Neon PostgreSQL | Durable structured system data | Large media binaries / real-time messaging |
| Android | Operator UX, local cache, verification/control requests | Direct Pi/MQTT/DB access |
| Evidence storage | Images/video/audio blobs | Core incident metadata |

This is a direct refinement of the previous implementation plan: the earlier "Gateway Database (PostgreSQL / SQLite)" becomes **Neon PostgreSQL as the canonical database**, while the Spring Boot gateway becomes the permanent backend boundary rather than something the Android app bypasses.

---

# 1. Cross-Check Against the Previous Implementation Plan

The existing plan already establishes most of the correct abstractions:

- REST + WebSocket hybrid for Android communication.
- Spring Boot Java gateway.
- MQTT between Pi and gateway.
- Room as an Android local cache.
- Separate AI prediction and human annotation records.
- Evidence delivered by reference rather than embedded binaries.
- JWT authentication.
- Autopilot policies stored centrally and synchronized to Pi.
- Incident state machine and audit trail.
- Incremental development.

Those decisions remain valid. The following changes are now required:

| Previous design | New design |
|---|---|
| Gateway DB = SQLite/PostgreSQL | **Neon PostgreSQL is the canonical database** |
| Gateway may live on Pi or VPS | **Backend is treated as a server-side service; Pi hosting is optional only for an isolated prototype** |
| Android talks to gateway | **Android talks only to backend HTTPS/WSS** |
| Pi sends to gateway | **Pi sends to backend through MQTT/HTTPS; backend persists into Neon** |
| Android caches gateway data | **Android caches backend data; Room is explicitly non-authoritative** |
| Evidence stored on gateway filesystem | **Metadata in Neon; binary evidence should eventually move to object storage** |
| Mock login allowed in DEBUG | **Remove mock authentication once the backend is wired; use a clearly isolated development seed/test path instead** |
| Build-time API URL | **Retain build-time configuration but add explicit dev/staging/production environments and a clear failure state** |
| Permanent WebSocket as background notification path | **WebSocket while active; FCM later for background/killed-app delivery** |

The old plan's state machine, schema concepts, MQTT topic hierarchy, training-data preservation rules and security principles should be retained rather than redesigned from scratch. The main architectural change is **centralization around the backend + Neon**.

---

# 2. Important Neon Credential Correction

The Android application must **never receive the Neon database credential or Neon management API key**.

Also, a Neon **API key is not the same thing as a PostgreSQL connection string**. Neon documents API keys for calls to the Neon management API, while application code connects to a database using a PostgreSQL connection string. A typical Neon connection string contains a PostgreSQL host, database user, password and database name, and can use a pooled endpoint. citeturn586680search1turn586680search8turn586680search6

Therefore:

```text
Android
   │
   │ HTTPS
   ▼
Spring Boot Backend
   │
   │ PostgreSQL connection
   ▼
Neon
```

**Never:**

```text
Android ──► Neon
Android ──► Neon API key
Android ──► PostgreSQL password
```

### Environment naming

Use the following in the backend rather than relying on a misleading `NEON_URL` name:

```dotenv
DATABASE_URL=<Neon PostgreSQL connection string>
JWT_SECRET=<long random secret>
```

Optional later variables:

```dotenv
MQTT_BROKER_URL=
MQTT_USERNAME=
MQTT_PASSWORD=
EVIDENCE_STORAGE_ENDPOINT=
EVIDENCE_STORAGE_BUCKET=
EVIDENCE_STORAGE_ACCESS_KEY=
EVIDENCE_STORAGE_SECRET_KEY=
```

If the existing `NEON_URL` value is an actual Neon API token rather than a PostgreSQL URL, **do not use it as `DATABASE_URL`**. Keep Neon management credentials separate from application database credentials.

If any secret has ever been committed to Git, rotate/revoke it rather than assuming `.gitignore` retroactively protects it.

Neon also supports pooled connection strings; using pooling is appropriate when the backend may open many short-lived database connections. citeturn586680search6turn586680search5

---

# 3. Scope of This Implementation Increment

The first implementation increment should be deliberately narrower than the full project.

## Build now

### Backend

1. Spring Boot backend project.
2. Neon PostgreSQL connectivity.
3. Flyway migrations.
4. User authentication.
5. JWT access + refresh tokens.
6. Incident REST API.
7. Node REST API.
8. Dashboard REST API.
9. Autopilot-policy REST API.
10. Incident verification API.
11. Audit-event API.
12. Backend health endpoint.
13. WebSocket event endpoint.
14. Deterministic API error format.
15. Server-side validation.
16. Unit + repository/integration tests.

### Android

1. Replace mock login with real backend login.
2. Point Retrofit to backend.
3. Make error reporting useful.
4. Fix WebSocket lifecycle/reconnection issues.
5. Fix session/refresh handling.
6. Make incident/node/cache synchronization backend-first.
7. Ensure Room is treated as cache only.
8. Add endpoint configuration for development/staging/production.
9. Preserve existing UI and domain model unless backend contract requires a correction.
10. Add tests against the backend contract.

## Do not implement yet

- Pi MQTT publisher/subscriber changes.
- Full evidence upload from Pi.
- FCM production delivery.
- Large media/object-storage infrastructure.
- Dataset export pipeline.
- Full node map UI.
- Full autopilot UI.
- Production multi-region/high-availability infrastructure.

Those become subsequent increments after Android ↔ backend is stable.

---

# 4. Backend Technology Stack

## Recommended stack

| Layer | Technology |
|---|---|
| Language | Java 17 |
| Framework | Spring Boot 4.1.x family |
| Security | Spring Security |
| Authentication | JWT |
| ORM | Spring Data JPA / Hibernate |
| Database | Neon PostgreSQL |
| Migration | Flyway |
| Web API | Spring MVC / REST |
| Realtime | WebSocket |
| MQTT | Eclipse Paho or Spring MQTT integration |
| Validation | Jakarta Bean Validation |
| JSON | Jackson |
| API docs | OpenAPI/Swagger |
| Testing | JUnit 5 + Mockito + Testcontainers |
| Build | Gradle |

Spring Boot currently lists 4.1.1 as a stable release, and Spring Security currently documents JWT bearer-token resource-server support. The exact patch version should be pinned when the backend repository is initialized. citeturn644246search15turn644246search0

Spring Boot provides first-class SQL/JPA integration, and Flyway migrations can execute on application startup from `classpath:db/migration`. citeturn586680search0turn586680search4

---

# 5. Proposed Backend Repository Structure

```text
sip-backend/
├── build.gradle
├── settings.gradle
├── Dockerfile
├── compose.yaml                     # local PostgreSQL/MQTT if desired
├── README.md
├── .gitignore
├── .env.example
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── MQTT.md
│   └── DEPLOYMENT.md
│
└── src/
    ├── main/
    │   ├── java/com/sip/backend/
    │   │   ├── SipBackendApplication.java
    │   │   │
    │   │   ├── config/
    │   │   │   ├── SecurityConfig.java
    │   │   │   ├── JwtConfig.java
    │   │   │   ├── WebSocketConfig.java
    │   │   │   ├── MqttConfig.java
    │   │   │   └── JacksonConfig.java
    │   │   │
    │   │   ├── auth/
    │   │   │   ├── AuthController.java
    │   │   │   ├── AuthService.java
    │   │   │   ├── JwtService.java
    │   │   │   ├── RefreshTokenService.java
    │   │   │   └── CustomUserDetailsService.java
    │   │   │
    │   │   ├── incident/
    │   │   │   ├── IncidentController.java
    │   │   │   ├── IncidentService.java
    │   │   │   ├── IncidentRepository.java
    │   │   │   ├── IncidentMapper.java
    │   │   │   └── IncidentStateMachine.java
    │   │   │
    │   │   ├── node/
    │   │   │   ├── NodeController.java
    │   │   │   ├── NodeService.java
    │   │   │   └── NodeRepository.java
    │   │   │
    │   │   ├── autopilot/
    │   │   │   ├── AutopilotController.java
    │   │   │   ├── AutopilotService.java
    │   │   │   └── AutopilotPolicyRepository.java
    │   │   │
    │   │   ├── dashboard/
    │   │   │   ├── DashboardController.java
    │   │   │   └── DashboardService.java
    │   │   │
    │   │   ├── evidence/
    │   │   │   ├── EvidenceController.java
    │   │   │   └── EvidenceService.java
    │   │   │
    │   │   ├── audit/
    │   │   │   ├── AuditService.java
    │   │   │   └── AuditRepository.java
    │   │   │
    │   │   ├── realtime/
    │   │   │   ├── WebSocketBroadcaster.java
    │   │   │   └── WebSocketEvent.java
    │   │   │
    │   │   ├── mqtt/
    │   │   │   ├── MqttSubscriber.java
    │   │   │   ├── MqttPublisher.java
    │   │   │   └── MqttMessageHandler.java
    │   │   │
    │   │   └── common/
    │   │       ├── ApiExceptionHandler.java
    │   │       ├── ApiError.java
    │   │       ├── PageResponse.java
    │   │       └── ClockProvider.java
    │   │
    │   └── resources/
    │       ├── application.yml
    │       ├── application-dev.yml
    │       └── db/
    │           └── migration/
    │               ├── V1__initial_schema.sql
    │               ├── V2__indexes.sql
    │               ├── V3__refresh_tokens.sql
    │               └── V4__audit_constraints.sql
    │
    └── test/
        └── java/com/sip/backend/
            ├── auth/
            ├── incident/
            ├── node/
            ├── autopilot/
            └── integration/
```

The structure is intentionally straightforward: controller → service → repository, with domain/state validation in the service/domain layer.

---

# 6. Neon Database Design

The previous SQL schema is a good conceptual starting point, but several implementation details should be corrected before creating the empty database schema.

## 6.1 Corrections to the previous schema

### Use `TIMESTAMPTZ`

All timestamps should be UTC-aware PostgreSQL timestamps.

```sql
TIMESTAMPTZ NOT NULL
```

Do not use timezone-less `TIMESTAMP` for event history.

### Create referenced tables first

The previous plan declared `incidents.node_id REFERENCES nodes(id)` before declaring `nodes` in the SQL block. The migration must create parent tables before child tables.

### Use PostgreSQL-native types

Prefer:

- `DOUBLE PRECISION` for coordinates and confidence.
- `BOOLEAN` for flags.
- `JSONB` for bounded flexible metadata/policy structures.
- `TEXT` for externally generated incident/node IDs.

### Treat incident IDs as idempotency keys

The Pi must be able to retry sending an incident without creating duplicates.

The backend should accept the Pi's stable incident ID and make creation idempotent.

---

# 7. Recommended Initial Schema

## 7.1 Users

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'OPERATOR',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_role_check
        CHECK (role IN ('ADMIN', 'OPERATOR'))
);
```

Passwords are stored only as strong password hashes. Never store plaintext operator passwords.

## 7.2 Nodes

```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'UNKNOWN',
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    last_heartbeat TIMESTAMPTZ,
    battery_level INTEGER,
    firmware_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT nodes_status_check
        CHECK (status IN ('ONLINE', 'DEGRADED', 'OFFLINE', 'UNKNOWN'))
);
```

## 7.3 Incidents

```sql
CREATE TABLE incidents (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    threat_type TEXT NOT NULL,
    threat_severity TEXT NOT NULL,
    threat_description TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location_readable TEXT,
    location_accuracy DOUBLE PRECISION,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    autopilot_handled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

The backend is authoritative for `state` transitions. The client does not arbitrarily set a new state.

## 7.4 Detection result

```sql
CREATE TABLE detection_results (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL UNIQUE REFERENCES incidents(id) ON DELETE CASCADE,
    predicted_class TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    detection_timestamp TIMESTAMPTZ NOT NULL,
    sensor_modalities JSONB,
    raw_scores JSONB
);
```

The original AI prediction is immutable from an application perspective. Human labels never overwrite this record.

## 7.5 Human annotations

```sql
CREATE TABLE human_annotations (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    annotator_id TEXT NOT NULL REFERENCES users(id),
    label TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    notes TEXT,
    confidence DOUBLE PRECISION,
    version INTEGER NOT NULL DEFAULT 1
);
```

Multiple annotations per incident remain supported, matching the original project requirement.

## 7.6 Evidence metadata

Instead of making one rigid "bundle" row with one path for each media type, use a flexible evidence table:

```sql
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    storage_key TEXT,
    mime_type TEXT,
    size_bytes BIGINT,
    sha256 TEXT,
    capture_timestamp TIMESTAMPTZ,
    retention_expiry TIMESTAMPTZ,
    upload_status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

This supports multiple images, multiple clips, thumbnails, future sensors and alternative evidence without changing the incident table.

## 7.7 Response events

```sql
CREATE TABLE response_events (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    result TEXT,
    autonomous BOOLEAN NOT NULL DEFAULT FALSE
);
```

## 7.8 Audit events

```sql
CREATE TABLE audit_events (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    description TEXT,
    actor TEXT
);
```

## 7.9 Autopilot policy

Keep common policy fields strongly typed, with flexible rule lists in JSONB:

```sql
CREATE TABLE autopilot_policies (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    allowed_threat_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.85,
    deterrence_timeout_seconds INTEGER NOT NULL DEFAULT 5,
    auto_escalate_on_deterrence_failure BOOLEAN NOT NULL DEFAULT TRUE,
    max_deterrence_attempts INTEGER NOT NULL DEFAULT 1,
    always_escalate_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    require_minimum_confidence BOOLEAN NOT NULL DEFAULT TRUE,
    require_multi_modal_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
    never_autonomous_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    sync_state TEXT NOT NULL DEFAULT 'DISABLED',
    last_sync TIMESTAMPTZ
);
```

## 7.10 Model versions

```sql
CREATE TABLE model_versions (
    version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    trained_at TIMESTAMPTZ,
    dataset_hash TEXT,
    validation_map50 DOUBLE PRECISION
);
```

---

# 8. Indexes

At minimum:

```sql
CREATE INDEX idx_incidents_created_at
    ON incidents (created_at DESC);

CREATE INDEX idx_incidents_state
    ON incidents (state);

CREATE INDEX idx_incidents_node_id
    ON incidents (node_id);

CREATE INDEX idx_incidents_threat_type
    ON incidents (threat_type);

CREATE INDEX idx_incidents_updated_at
    ON incidents (updated_at DESC);

CREATE INDEX idx_annotations_incident_id
    ON human_annotations (incident_id);

CREATE INDEX idx_response_events_incident_id
    ON response_events (incident_id);

CREATE INDEX idx_audit_events_incident_id
    ON audit_events (incident_id);

CREATE INDEX idx_audit_events_timestamp
    ON audit_events (timestamp DESC);
```

These support the Android feed, incident detail, dashboard and history queries.

---

# 9. REST API Contract

The current Android app already expects these endpoint shapes. The backend should implement **the existing contract first**, rather than forcing a large Android rewrite.

## Authentication

```http
POST /api/v1/auth/login
POST /api/v1/auth/refresh
POST /api/v1/auth/logout
```

Login request:

```json
{
  "username": "admin",
  "password": "..."
}
```

Login response:

```json
{
  "token": "...",
  "refreshToken": "...",
  "expiresIn": 900,
  "userId": "...",
  "username": "admin",
  "role": "ADMIN"
}
```

Recommended access-token lifetime: roughly 10–15 minutes. Refresh tokens should be longer-lived and revocable.

## Incidents

```http
GET  /api/v1/incidents?page=0&size=50
GET  /api/v1/incidents/{id}
POST /api/v1/incidents/{id}/verify
GET  /api/v1/incidents/{id}/events
```

The existing Android `PageDto` expects:

```json
{
  "content": [],
  "page": 0,
  "size": 50,
  "totalElements": 123,
  "totalPages": 3
}
```

The backend should return this exact shape in the initial integration.

## Nodes

```http
GET /api/v1/nodes
GET /api/v1/nodes/{id}
```

## Dashboard

```http
GET /api/v1/dashboard
```

Response shape compatible with the current Android DTO:

```json
{
  "totalNodes": 1,
  "onlineNodes": 1,
  "activeThreats": 0,
  "recentIncidents": [],
  "systemHealth": "GREEN"
}
```

## Autopilot

```http
GET /api/v1/autopilot/policy?nodeId=<nodeId>
PUT /api/v1/autopilot/policy?nodeId=<nodeId>
```

## Evidence

```http
GET /api/v1/evidence/{incidentId}/image
GET /api/v1/evidence/{incidentId}/video
GET /api/v1/evidence/{incidentId}/audio
GET /api/v1/evidence/{incidentId}/thumbnail
```

A later Pi integration increment adds:

```http
POST /api/v1/evidence/upload
```

with authenticated multipart upload.

---

# 10. API Error Contract

The current Android login layer is too generic: almost every unexpected exception becomes `Network error. Check backend URL.`

Replace that with a stable JSON error format:

```json
{
  "timestamp": "2026-09-30T17:30:00Z",
  "status": 401,
  "code": "AUTH_INVALID_CREDENTIALS",
  "message": "Invalid username or password",
  "path": "/api/v1/auth/login",
  "requestId": "..."
}
```

Suggested error codes:

| Code | Meaning |
|---|---|
| `AUTH_INVALID_CREDENTIALS` | Login failed |
| `AUTH_SESSION_EXPIRED` | Refresh token expired/revoked |
| `AUTH_FORBIDDEN` | User lacks permission |
| `INCIDENT_NOT_FOUND` | Unknown incident |
| `INVALID_STATE_TRANSITION` | Requested state transition is illegal |
| `VALIDATION_ERROR` | Request body/query invalid |
| `NODE_NOT_FOUND` | Unknown node |
| `BACKEND_UNAVAILABLE` | Dependency unavailable |
| `INTERNAL_ERROR` | Unexpected server failure |

Android should map these to human-readable UI messages.

---

# 11. Authentication Design

## 11.1 Login

```text
Android
  │ POST /auth/login
  ▼
Backend
  │ lookup username
  │ verify password hash
  │ issue access + refresh token
  ▼
Android SecureTokenStore
```

## 11.2 Authorization

Roles:

```text
ADMIN
OPERATOR
```

Initial permissions:

| Operation | ADMIN | OPERATOR |
|---|---:|---:|
| View dashboard | ✓ | ✓ |
| View incidents | ✓ | ✓ |
| Verify/reject incident | ✓ | ✓ |
| Trigger manual deterrence | ✓ | ✓ |
| Change autopilot policy | ✓ | optionally restricted |
| Manage users | ✓ | ✗ |
| Administrative node configuration | ✓ | ✗ |

The exact operational permissions can be refined later, but the role boundary should exist in the backend now.

## 11.3 Refresh tokens

Store refresh tokens server-side in a hashed/revocable form rather than treating a long-lived opaque token as permanently valid.

Recommended table:

```sql
CREATE TABLE refresh_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

The backend should rotate refresh tokens on use.

---

# 12. Fixes Required in the Current Android App

The latest `master` build is already CI-green, so the following are **runtime/correctness hardening tasks**, not compiler repairs.

## Bug 1 — Physical-device API URL mismatch

Current build configuration defaults to:

```text
http://10.0.2.2:8080/
```

That address is appropriate for an Android emulator reaching the host machine; it is not the correct address for a physical phone.

This is the first thing that explains the screenshot's backend-URL error path when the phone is running an APK built with the local/emulator default.

### Fix

Introduce explicit environments:

```text
local-emulator
local-lan
staging
production
```

Example concept:

```text
local-emulator → http://10.0.2.2:8080/
local-lan      → http://<developer-machine-lan-ip>:8080/
staging        → https://api-staging.<domain>/
production     → https://api.<domain>/
```

Release builds must not silently fall back to `10.0.2.2`.

---

## Bug 2 — DEBUG mock login bypasses the real backend

`AuthRepositoryImpl` currently contains hard-coded DEBUG credentials and bypasses the API for those users.

That was useful for the original UI prototype, but it becomes harmful now because it can make the app appear authenticated even when the real backend contract is broken.

### Fix

Remove the default mock-login path from the normal application flow.

For development tests, use one of these instead:

- backend seed user,
- MockWebServer tests,
- a dedicated `fake` build flavor that cannot be confused with staging/production.

Do not keep production-like credentials in application source.

---

## Bug 3 — Login errors are too generic

Current behavior collapses unrelated failures into:

```text
Network error. Check backend URL.
```

### Fix

Handle separately:

- DNS/connection refused → `Cannot reach backend`.
- timeout → `Backend timed out`.
- TLS failure → `Secure connection failed`.
- HTTP 401 → `Invalid username/password`.
- HTTP 403 → `Account is not authorized`.
- HTTP 5xx → `Backend service error`.
- malformed JSON → `Backend returned an invalid response`.

This will make field debugging dramatically easier.

---

## Bug 4 — WebSocket duplicate-connect race

`SipWebSocketClient.connect()` can open another socket when called while a previous connection/reconnect cycle is still active.

### Fix

Add an explicit connection state machine:

```text
DISCONNECTED
CONNECTING
CONNECTED
RECONNECT_WAIT
STOPPING
```

Only one active socket attempt may exist at a time.

Store a monotonically increasing connection generation/token so callbacks from stale sockets cannot schedule a second reconnect after a new socket has already connected.

---

## Bug 5 — WebSocket token-expiry handling

The current client reads the current access token from `SecureTokenStore`; when the access token expires, the WebSocket may keep retrying without first ensuring a valid token exists.

### Fix

Before reconnecting:

1. Detect missing/expired access token.
2. Attempt refresh-session logic once.
3. Re-read the new access token.
4. Reconnect.
5. On refresh failure, clear the session and propagate a logged-out state to the app.

The WebSocket should never reconnect indefinitely with an unusable token.

---

## Bug 6 — WebSocket URL construction is too naive

The current code derives the WebSocket URL with string replacement of `http://` / `https://`.

### Fix

Parse the REST base URL as a proper `HttpUrl`, then construct:

```text
ws://host[:port]/ws/events
```

or

```text
wss://host[:port]/ws/events
```

This avoids path, trailing-slash, query and port edge cases.

---

## Bug 7 — WebSocket listener does synchronous database work

The WebSocket callback path currently receives an incident and immediately calls a Room DAO operation.

### Fix

Offload persistence to a background executor/coroutine and keep WebSocket callbacks lightweight.

```text
WebSocket callback
      ↓
parse event
      ↓
queue persistence
      ↓
Room
      ↓
UI/notification
```

---

## Bug 8 — Token refresh should synchronize stored user metadata

The refresh authenticator refreshes tokens but does not necessarily refresh persisted user metadata.

### Fix

If refresh responses contain user identity/role, persist the corresponding session metadata atomically with the new token set.

---

## Bug 9 — Unknown backend roles currently fail open to OPERATOR semantics

The current `currentUser()` logic treats anything not equal to `ADMIN` as `OPERATOR`.

### Fix

Use explicit role parsing and fail closed:

```text
ADMIN → ADMIN
OPERATOR → OPERATOR
anything else → invalid session / logout
```

---

## Bug 10 — Autopilot mapper has unsafe enum parsing

Some fields in `AutopilotMapper` use direct `Enum.valueOf()` while other fields use safe parsing.

### Fix

Apply the same safe-enum strategy consistently to:

- allowed threat types,
- always-escalate types,
- never-autonomous types,
- sync state.

---

## Bug 11 — Node timestamp parsing is not consistently defensive

The node repository directly parses some timestamps with `Instant.parse()`.

### Fix

Use one shared safe time parser and treat malformed timestamps as invalid data rather than crashing the repository.

---

## Bug 12 — `getNodeById()` should not fetch the entire node collection

The current implementation can retrieve all nodes and then select one.

### Fix

Use the existing dedicated endpoint:

```http
GET /api/v1/nodes/{id}
```

and fall back to the local Room row only when the network is unavailable.

---

## Bug 13 — `observeIncidentById()` must handle a missing Room row safely

A missing row may produce a null value before the mapping layer.

### Fix

Represent absence explicitly (`Maybe`, nullable mapping, or a filtered stream) instead of blindly calling the mapper.

---

## Bug 14 — Session state is not globally reactive

`MainActivity` derives an `authenticated` boolean once during `onCreate()`.

### Fix

Introduce a session state source, for example:

```text
SessionState = LOGGED_OUT | AUTHENTICATED | REFRESHING | EXPIRED
```

The UI should react to session expiration instead of assuming the original authentication state remains valid forever.

---

## Bug 15 — Notification deep-linking is incomplete

The local notification currently opens `MainActivity`, but a notification for incident `X` should open the corresponding incident detail screen.

### Fix

Put the incident ID in the notification intent and route directly to:

```text
IncidentDetail/{incidentId}
```

If the incident has already been deleted/expired, fall back to the incident feed.

---

## Bug 16 — Foreground WebSocket service is not a final notification architecture

The existing service is useful for active monitoring, but the project plan already recognizes that Android background/killed-app delivery cannot depend on a permanent socket.

Android 15 also limits the runtime budget of a `dataSync` foreground service, so this is not a permanent background-monitoring mechanism.

### Fix

Short-term:

```text
App active → WebSocket
App resumed → REST reconciliation
```

Next increment:

```text
Background/killed → FCM notification
App opened → REST reconciliation
```

---

# 13. Android Network Architecture After Migration

Target architecture:

```text
Compose UI
   ↓
ViewModel (Kotlin)
   ↓
Use Case (Java)
   ↓
Repository (Java)
   ├── Retrofit REST client
   ├── WebSocket client
   └── Room cache
```

Backend is the only remote dependency.

The Android app should know **only**:

```text
API_BASE_URL
```

It should not know:

```text
Neon host
Neon password
Neon API key
MQTT broker credentials
Pi SSH credentials
Pi database credentials
```

---

# 14. Android API Configuration Strategy

The current build-property mechanism is useful and should be retained, but it needs explicit environments.

## Suggested Gradle flavors

```text
dev
staging
prod
```

Conceptual URLs:

```text
dev     → local backend
staging → cloud staging backend
prod    → cloud production backend
```

The production base URL becomes a normal application endpoint, not a secret.

Only secrets remain server-side.

---

# 15. Backend State Ownership Rules

This is critical.

## Android may request

```text
VERIFY incident
REJECT incident
TRIGGER deterrence
UPDATE autopilot policy
```

## Android may not decide

```text
DETECTED → VERIFIED
```

by changing a local field and assuming the operation succeeded.

Instead:

```text
Android
  │ POST verification
  ▼
Backend
  │ validates permission
  │ locks/loads incident
  │ validates state transition
  │ creates HumanAnnotation
  │ writes AuditEvent
  │ updates Incident
  │ commits transaction
  │ publishes WebSocket event
  ▼
Android
  │ receives authoritative result
  ▼
Room cache
```

The server therefore becomes the source of truth for incident lifecycle transitions.

---

# 16. Incident Ingestion Contract for Future Pi Integration

The Pi should eventually publish a complete event that the backend can persist without the Android app being involved.

Example logical payload:

```json
{
  "id": "pi-node-01-20260930-000123",
  "nodeId": "node-01",
  "state": "EVIDENCE_CAPTURED",
  "threat": {
    "type": "WEAPON",
    "severity": "NON_DETERRABLE",
    "description": "Potential weapon detected"
  },
  "location": {
    "latitude": 0.0,
    "longitude": 0.0,
    "humanReadable": "...",
    "accuracy": 5.0
  },
  "detection": {
    "predictedClass": "weapon",
    "confidence": 0.94,
    "modelVersion": "weapon-yolov8n-v3",
    "detectionTimestamp": "2026-09-30T17:00:00Z",
    "sensorModalities": ["CAMERA", "PIR"]
  },
  "autopilotHandled": false,
  "created_at": "2026-09-30T17:00:00Z"
}
```

The backend validates the schema, creates the incident idempotently and emits `INCIDENT_NEW` to connected Android clients.

---

# 17. Idempotency and Duplicate Protection

This should be implemented in the backend from day one.

Suppose a Pi publishes an incident and the network retries:

```text
Publish #1 → backend creates incident
Publish #2 → same incident ID
Publish #3 → same incident ID
```

The result must still be:

```text
1 incident
```

not:

```text
3 incidents
```

Implementation options:

- Primary-key uniqueness on incident ID.
- Transactional insert.
- MQTT QoS 1.
- Optional ingestion request/idempotency table if REST ingestion is also exposed.

---

# 18. WebSocket Event Architecture

Backend publishes events only after the database transaction succeeds.

Sequence:

```text
REST/MQTT event
      ↓
validate
      ↓
DB transaction
      ↓
commit
      ↓
publish WebSocket event
```

Never publish a "VERIFIED" event before the DB transaction commits.

## Event envelope

```json
{
  "type": "INCIDENT_NEW",
  "timestamp": "2026-09-30T17:05:00Z",
  "payload": { ... },
  "eventId": "..."
}
```

`eventId` allows Android to deduplicate events if a future delivery system retries them.

---

# 19. WebSocket Connection Model

Endpoint:

```text
wss://api.example.com/ws/events
```

Authentication must use a secure handshake mechanism. Do not put bearer tokens in query parameters because URLs can appear in logs and diagnostics.

The previous project plan already corrected this requirement; keep that decision.

Client behavior:

```text
CONNECTING
    ↓
CONNECTED
    ↓
DISCONNECTED
    ↓
RECONNECT_WAIT
    ↓
CONNECTING
```

Use exponential backoff with jitter and a hard maximum.

---

# 20. REST + WebSocket Synchronization Rule

The two channels have different purposes:

| REST | WebSocket |
|---|---|
| Initial screen load | Live updates |
| Historical queries | Incident created |
| Filters/pagination | Incident state changed |
| Verification commands | Node status changed |
| Policy configuration | Policy sync event |
| Reconciliation after reconnect | System alert |

The WebSocket is **not** a durable database.

When a client reconnects:

```text
1. REST fetch current data
2. Replace/reconcile stale Room state
3. Start WebSocket
4. Resume live events
```

This prevents event loss during network interruptions.

---

# 21. Evidence Architecture

Do not use Neon as a binary file store.

Use:

```text
Neon
  └── evidence metadata

Object storage
  ├── image
  ├── thumbnail
  ├── video
  └── audio
```

Evidence records contain:

```text
storage_key
mime_type
size_bytes
sha256
capture_timestamp
retention_expiry
```

The backend provides authenticated URLs/endpoints to Android.

For the current app-backend increment, evidence can remain **metadata-only** until the incident CRUD contract is stable.

The previous implementation plan's reference-based evidence strategy remains correct.

---

# 22. MQTT Architecture for the Next Increment

Once Android ↔ backend works, integrate Pi:

```text
Pi
 └── MQTT publish
        ↓
Mosquitto
        ↓
Spring Boot MQTT subscriber
        ↓
IncidentService / NodeService
        ↓
Neon
        ↓
WebSocket
        ↓
Android
```

### Pi → Backend topics

```text
sip/incident/new
sip/incident/{incidentId}/update
sip/node/{nodeId}/heartbeat
sip/node/{nodeId}/status
sip/node/{nodeId}/metrics
sip/evidence/{incidentId}
```

### Backend → Pi topics

```text
sip/command/{nodeId}/verify
sip/command/{nodeId}/deterrence
sip/command/{nodeId}/policy
```

Keep QoS choices from the previous plan:

| Topic | QoS |
|---|---:|
| Incident | 1 |
| Command | 1 |
| Heartbeat | 0 |
| Evidence metadata | 1 |

---

# 23. Autopilot Safety Boundary

The backend stores and distributes policy.

The Pi remains autonomous.

Correct flow:

```text
Android
  ↓ policy update
Backend
  ↓ persist policy in Neon
  ↓ publish MQTT policy
Pi
  ↓ validate policy
  ↓ store last-known-good local policy
  ↓ execute autonomy locally
```

The Pi must not require Android or Neon to be reachable for every autonomous deterrence action.

This preserves the safety requirement in the original project architecture.

---

# 24. Backend Validation Rules

Server-side validation must cover:

### Incidents

- non-empty ID;
- known node;
- known state;
- confidence in `[0,1]`;
- valid timestamps;
- valid threat types/severities;
- immutable original detection record.

### Verification

- incident exists;
- current state is `PENDING_VERIFICATION`;
- operator is authenticated;
- operator role is authorized;
- annotation label is valid;
- transition is legal.

### Autopilot

- confidence threshold within bounds;
- timeout positive and bounded;
- max attempts positive and bounded;
- safety rules cannot be silently omitted;
- policy changes are audited.

---

# 25. Transaction Boundaries

Each important operator action should be one backend transaction.

### Verify example

```text
BEGIN
  load incident
  validate current state
  create human annotation
  update incident state
  create audit event
  create response event if applicable
COMMIT
publish WebSocket event
```

If any DB operation fails:

```text
ROLLBACK
no success event is sent
```

This is significantly safer than allowing the Android app to mutate separate local objects independently.

---

# 26. Android Room Synchronization Strategy

Room remains useful, but it is no longer the source of truth.

## On first login

```text
Backend → fetch dashboard/nodes/incidents
       → Room cache
       → UI
```

## While connected

```text
WebSocket event
   ↓
update Room
   ↓
UI observes Room
```

## On reconnect

```text
REST reconciliation
   ↓
Room upsert
   ↓
WebSocket resume
```

## On write

Do not optimistically invent a successful backend state.

Example:

```text
Tap VERIFY
   ↓
loading state
   ↓
POST /verify
   ↓
backend result
   ↓
Room update
   ↓
UI update
```

A future offline command queue can be added later, but not before the online path is correct.

---

# 27. Testing Strategy

## 27.1 Backend unit tests

### Authentication

- valid login;
- wrong password;
- disabled account;
- expired refresh token;
- revoked refresh token;
- refresh rotation.

### Incident state machine

Test every valid and invalid transition from the original plan.

### Validation

- invalid confidence;
- unknown threat type;
- nonexistent node;
- malformed timestamps;
- illegal verification state.

### Service behavior

- duplicate incident ID;
- annotation creation;
- audit creation;
- node heartbeat update.

## 27.2 Backend integration tests

Use PostgreSQL/Testcontainers or a dedicated test database.

Test:

```text
POST login
GET incidents
GET incident
POST verify
GET events
GET dashboard
GET nodes
GET policy
PUT policy
```

## 27.3 Android repository tests

MockWebServer should verify the exact backend JSON contract.

Test:

- 200 responses;
- 401 + refresh;
- 403;
- 404;
- 409;
- 422 validation error;
- 500;
- timeout;
- invalid JSON;
- empty page;
- pagination.

## 27.4 WebSocket tests

Test:

```text
connect
→ authenticate
→ receive INCIDENT_NEW
→ update Room
→ disconnect
→ reconnect
→ REST reconcile
→ receive new events
```

Also test stale socket callbacks and duplicate connection attempts.

---

# 28. API Contract Testing

The Android and backend projects should share one contract document.

Recommended:

```text
SIP-Backend/docs/openapi.yaml
```

Generate Swagger/OpenAPI documentation from the backend, then make Android DTO/service changes against that contract.

The goal is to eliminate silent mismatches such as:

```text
Android expects: created_at
Backend sends:   createdAt
```

or

```text
Android expects PageDto.content
Backend sends content as an object
```

The contract becomes the source of truth for transport models.

---

# 29. Development Deployment

## Local development

```text
Fedora laptop
├── SIP-Backend (Spring Boot)
└── Android Studio / Gradle

Neon
└── development branch/database
```

Android physical device:

```text
Phone ──Wi-Fi──► Laptop LAN IP ──► Spring Boot ──► Neon
```

Android emulator:

```text
Emulator ──► 10.0.2.2:8080 ──► Spring Boot ──► Neon
```

This distinction directly prevents the screenshot's current URL problem.

---

# 30. Neon Branching Strategy

Use separate Neon branches for development/testing when practical:

```text
production
├── staging
└── development
```

Neon branches provide isolated Postgres environments with separate connection strings, which is useful for testing migrations without touching production data. citeturn586680search11

Do not let automated Android tests write against production Neon.

---

# 31. Backend Deployment

## Phase A — local

Run the Spring Boot server directly on the development machine.

## Phase B — staging

Deploy the backend to an always-accessible HTTPS endpoint and connect it to a Neon development/staging branch.

For quick proof-of-concept testing, a free Render web service is possible, but Render documents that free web services spin down after 15 minutes without inbound traffic and can take about a minute to start again. That makes the free tier suitable for development/demo HTTP testing, not for a final always-on realtime safety backend. citeturn133820search1turn133820search2

## Phase C — production

Use an always-on backend host/VPS/container platform.

Desired production topology:

```text
HTTPS/WSS
    ↓
Spring Boot backend
    ├── Neon PostgreSQL
    ├── MQTT broker
    └── Object storage
```

The production host should not rely on a sleeping/free compute instance for critical realtime delivery.

---

# 32. Health and Observability

Implement:

```http
GET /actuator/health
```

and expose only safe health information publicly.

Log:

- request ID;
- authenticated user ID (not passwords/tokens);
- endpoint;
- response status;
- duration;
- incident ID/node ID when relevant;
- backend exceptions.

Never log:

- passwords;
- refresh tokens;
- access tokens;
- Neon connection strings;
- MQTT passwords;
- Pi SSH credentials.

---

# 33. Security Checklist

### Backend

- HTTPS in staging/production.
- JWT access tokens.
- Rotating refresh tokens.
- Password hashing.
- Role-based authorization.
- Input validation.
- Server-side state validation.
- Request IDs.
- Audit events.
- Rate limiting on login.
- No secret values in source.

### Android

- HTTPS/WSS only in staging/production.
- Encrypted token storage.
- No DB credentials.
- No MQTT credentials.
- No Pi SSH credentials.
- No Neon API key.
- No backend database URL.

Spring Security's current documentation supports JWT bearer-token validation and issuer/JWK-based validation for resource-server-style security, providing a standard foundation for the backend authentication layer. citeturn644246search0turn644246search11

---

# 34. Implementation Roadmap

## Increment A — Backend shell

**Goal:** backend starts and exposes health endpoint.

Tasks:

- initialize Spring Boot project;
- Java 17;
- Gradle;
- configuration profiles;
- `/actuator/health`;
- global error handler;
- structured logging.

**Deliverable:** running backend.

---

## Increment B — Neon + Flyway

**Goal:** backend writes to the empty Neon database.

Tasks:

- configure `DATABASE_URL`;
- connect to Neon;
- add Flyway;
- create V1 schema;
- create V2 indexes;
- run migration against development branch;
- verify tables in Neon.

**Deliverable:** actual persistent backend database.

---

## Increment C — Authentication

**Goal:** the Android login screen can authenticate against the backend.

Tasks:

- `users` table;
- admin seed process for development;
- password hashing;
- login endpoint;
- JWT issuance;
- refresh token rotation;
- role claims;
- `/auth/refresh`;
- Android mock login removal.

**Deliverable:** real login.

---

## Increment D — Incidents + Nodes

**Goal:** Android reads real DB data.

Tasks:

- repositories/entities;
- pagination;
- filters;
- incident detail;
- nodes;
- dashboard queries;
- deterministic error contract.

**Deliverable:** Dashboard + Incident Feed backed by Neon.

---

## Increment E — Android runtime hardening

**Goal:** fix the bugs identified from the current repository.

Tasks:

- environment-specific API URL;
- physical-device testing;
- real login;
- HTTP error mapping;
- session state;
- token refresh hardening;
- WebSocket duplicate connection fix;
- WebSocket token refresh;
- safe URL construction;
- Room background persistence;
- safe enum/time parsing;
- node endpoint optimization;
- notification deep link.

**Deliverable:** stable Android client.

---

## Increment F — Verification + audit

**Goal:** operator writes authoritative changes through the backend.

Tasks:

- `POST /incidents/{id}/verify`;
- backend state machine;
- human annotation persistence;
- audit event persistence;
- response event persistence;
- WebSocket broadcast after commit.

**Deliverable:** complete operator verification loop.

---

## Increment G — WebSocket

**Goal:** real-time Android updates.

Tasks:

- backend WS endpoint;
- authentication;
- event envelopes;
- broadcaster;
- Android reconnection;
- REST reconciliation after reconnect.

**Deliverable:** live incident feed.

---

## Increment H — Autopilot API

**Goal:** policy storage and control-plane support.

Tasks:

- GET/PUT policy;
- server validation;
- audit policy changes;
- WebSocket `POLICY_SYNCED` event contract;
- keep actual autonomous execution local to Pi.

**Deliverable:** backend-ready autopilot management.

---

## Increment I — Pi integration

**Goal:** Pi becomes a backend client instead of an Android data source.

Tasks:

- MQTT publisher;
- heartbeat publisher;
- incident publisher;
- status publisher;
- backend MQTT subscriber;
- evidence upload;
- retry/idempotency;
- command topics;
- local last-known-good policy.

**Deliverable:**

```text
Pi → Backend → Neon
              ↓
           Android
```

---

## Increment J — Evidence and FCM

**Goal:** field-ready operational behavior.

Tasks:

- object storage;
- authenticated evidence retrieval;
- retention;
- thumbnails;
- video range requests;
- FCM;
- background/killed-app notification flow.

**Deliverable:** production-grade incident evidence + notifications.

---

# 35. Definition of Done for the Immediate Backend Phase

Do not move on to Pi integration until all of the following are true:

### Backend

- [ ] Spring Boot starts cleanly.
- [ ] Neon connection works.
- [ ] Flyway creates the schema.
- [ ] Admin user can be seeded safely.
- [ ] Login works.
- [ ] Refresh works.
- [ ] Invalid login returns HTTP 401 + structured error.
- [ ] Incidents can be created/queried.
- [ ] Nodes can be queried.
- [ ] Dashboard works.
- [ ] Verification is transactional.
- [ ] Audit event is written.
- [ ] Duplicate incident IDs are idempotent.
- [ ] WebSocket events are emitted after commit.

### Android

- [ ] No DEBUG mock login in the normal build.
- [ ] Physical phone reaches backend.
- [ ] Login succeeds against backend.
- [ ] Dashboard loads from backend.
- [ ] Incident feed loads from backend.
- [ ] Room caches successful backend data.
- [ ] Offline cached data remains viewable.
- [ ] 401 refresh path works.
- [ ] Session expiry logs out cleanly.
- [ ] WebSocket reconnect is single-instance and token-aware.
- [ ] Notifications deep-link to incidents.
- [ ] No Neon/MQTT/Pi secrets are in APK/source.

---

# 36. Final Target Architecture

After all increments, the system should converge to:

```text
                        ┌────────────────────────┐
                        │      Neon PostgreSQL    │
                        │  Canonical data store   │
                        └───────────▲────────────┘
                                    │
                         ┌──────────┴───────────┐
                         │     SIP Backend      │
                         │     Spring Boot      │
                         │                      │
                         │ REST │ WebSocket     │
                         │ Auth │ Audit         │
                         │ MQTT │ Evidence API  │
                         └──────▲──────┬────────┘
                                │      │
                         MQTT/HTTPS   HTTPS/WSS
                                │      │
                  ┌─────────────┘      └─────────────┐
                  │                                  │
          ┌───────┴────────┐                 ┌───────┴─────────┐
          │ Raspberry Pi 5 │                 │ SIP Guardian    │
          │ Edge AI Node   │                 │ Android App     │
          │                │                 │                 │
          │ sensors        │                 │ Compose UI      │
          │ YOLO/audio     │                 │ Room cache      │
          │ deterrence     │                 │ verification    │
          │ local policy   │                 │ control         │
          └────────────────┘                 └─────────────────┘
```

### The key invariant

```text
Pi does AI.
Backend owns operational truth.
Neon stores durable structured data.
Android operates the system.
```

Android should therefore never need to know where the Raspberry Pi is, what its IP address is, how MQTT is configured, or where Neon is hosted. It only needs the authenticated backend URL.

---

# 37. Immediate Execution Order

The implementation should be performed in exactly this order:

```text
1. Create SIP-Backend repo
2. Create Spring Boot project
3. Configure environment variables
4. Connect to Neon
5. Create Flyway V1 schema
6. Verify tables in Neon
7. Implement user/authentication
8. Implement incidents/nodes/dashboard
9. Add structured API errors
10. Point Android at backend
11. Remove mock login path
12. Fix Android runtime/network/session bugs
13. Add verification + audit
14. Add WebSocket
15. Run Android ↔ backend integration tests
16. Only then touch Raspberry Pi integration
```

This sequencing avoids repeating the earlier problem where the Android app was built around a gateway contract that had no real server behind it.

---

## Source Cross-Check

This plan was cross-checked against:

- the existing Android implementation plan, particularly the architecture, API, WebSocket, MQTT, evidence, database, security, testing, roadmap and deployment sections;
- the updated `CONTEXT.md`, particularly the settled Raspberry Pi 5 architecture, autonomous edge-compute requirement, training-data lifecycle and current project next steps;
- the current `master` branch of `bchbenjamin/SIP-Android`, including the Retrofit API contract, authentication repository, network module, WebSocket client/service, Room repositories and DTOs;
- current Neon documentation for application connection strings, API keys and connection pooling;
- current Spring documentation for Spring Boot/Spring Security/JWT and Flyway;
- current Render documentation for free-tier deployment limitations.

**Repository modification status:** this planning pass does not modify `SIP-Android`.
