# SIP Backend

Spring Boot 3 / Java 17 backend for the AI-Powered Street Safety Device Network.

The backend owns authentication, REST APIs, WebSocket event delivery, audit records, and PostgreSQL persistence. Android communicates only with this backend; Neon is never accessed directly by the APK.

## Requirements

- JDK 17+
- Gradle 8.9+ (or the included Gradle wrapper)
- Neon PostgreSQL or a local PostgreSQL database

## Configure

Copy `.env.example` to a private `.env`, then export its values into the process environment. Spring Boot and Gradle do not automatically load a dotenv file.

- `DATABASE_URL`: full PostgreSQL JDBC URL. For Neon, use the JDBC URL and TLS settings from Neon connection details; do not use a Neon management API key here.
- `JWT_SECRET`: strong random secret (at least 32 bytes). Generate one with `openssl rand -base64 48`.
- `SIP_BOOTSTRAP_ADMIN_USERNAME` and `SIP_BOOTSTRAP_ADMIN_PASSWORD`: optional first-admin bootstrap. Set both only for initial provisioning; password must be at least 12 characters. No default credentials are provided.

Example local shell:

```bash
export DATABASE_URL='jdbc:postgresql://localhost:5432/sipdb?user=sipuser'
export JWT_SECRET="$(openssl rand -base64 48)"
./gradlew bootRun
```

Flyway migrations run at startup. Hibernate schema auto-update is disabled; schema changes must be added as new migrations.

## Useful commands

```bash
./gradlew clean test
./gradlew bootJar
java -jar build/libs/sip-backend-0.1.0.jar
curl http://localhost:8080/actuator/health
```

## API overview

- `POST /api/v1/auth/login` — obtain access and refresh tokens.
- `POST /api/v1/auth/refresh` — rotate a refresh token.
- `POST /api/v1/auth/logout` — revoke refresh sessions for the authenticated user.
- `GET /api/v1/incidents`, `GET /api/v1/incidents/{id}` — incident feed/details.
- `POST /api/v1/incidents/{id}/verify` — record an operator annotation and lifecycle change.
- `GET /api/v1/incidents/{id}/events` — audit events.
- `GET /api/v1/nodes`, `GET /api/v1/nodes/{id}` — node status.
- `GET /api/v1/dashboard` — dashboard summary.
- `GET /api/v1/autopilot/policy?nodeId=...`, `PUT /api/v1/autopilot/policy?nodeId=...` — policy management.
- `/ws/events` — authenticated WebSocket events (bearer access token required in handshake header).

## Known integration boundaries

The backend currently has no complete MQTT ingestion/command bridge or Pi publisher integration. Evidence metadata endpoints are not yet a production object-storage pipeline. See `../docs/IMPLEMENTATION_STATUS.md`; do not assume those paths are production-ready.
