# SIP Guardian — AI-Powered Street Safety Device Network

This repository is a monorepo for the Android operator app and the Spring Boot backend.

## Repository layout

- `android/` — Android client (Java domain/data layers, Kotlin + Jetpack Compose UI).
- `backend/` — Spring Boot REST/WebSocket API, authentication, Flyway migrations and Neon/PostgreSQL integration.
- `CONTEXT/` — project context and the consolidated implementation plan.
- `docs/IMPLEMENTATION_STATUS.md` — implemented capabilities, known gaps and verification status.\n- `docs/IMPLEMENTATION_PLAN_DELTA.md` — current monorepo decisions and implementation gaps; it supersedes conflicting repository instructions in the original plan.

## Architecture

```text
Raspberry Pi edge nodes (local inference/autonomy)
            │ future MQTT/HTTPS integration
            ▼
Spring Boot backend ─────── Neon PostgreSQL
            ▲
            │ HTTPS / WSS
            ▼
SIP Guardian Android app ── Room local cache
```

The backend is the canonical API and database boundary. Android must not connect directly to Neon, MQTT or Raspberry Pi nodes. Room is an offline cache, not the system of record. Edge autonomy must continue locally if the network or backend is unavailable.

## Backend setup

Requirements: JDK 17+, Gradle 8.9+, and a PostgreSQL database (Neon recommended).

1. Copy `backend/.env.example` to a private `backend/.env` and configure `DATABASE_URL` and `JWT_SECRET`. The backend does not read a Neon management API key as a database URL.
2. Export those environment variables in your shell (Gradle/Spring Boot does not automatically load a `.env` file).
3. Start the backend:

   ```bash
   cd backend
   ./gradlew bootRun
   ```

4. Check health: `curl http://localhost:8080/actuator/health`.

To create the first administrator, set both `SIP_BOOTSTRAP_ADMIN_USERNAME` and `SIP_BOOTSTRAP_ADMIN_PASSWORD` before first startup. Use a unique password of at least 12 characters. There are no intended default login credentials.

## Android setup

Set the API base URL when building. For an emulator, `http://10.0.2.2:8080/` reaches the host machine. For a physical phone, use the host's LAN address during development or a deployed HTTPS URL.

```bash
cd android
./gradlew clean testDebugUnitTest assembleDebug -PSIP_API_BASE_URL=http://10.0.2.2:8080/
```

The debug APK is created at `android/app/build/outputs/apk/debug/app-debug.apk`.

## Security notes

- Never put Neon credentials, backend secrets, Pi SSH credentials or node credentials in Android resources, BuildConfig, the APK or version control.
- Use HTTPS/WSS outside local development.
- Configure a strong random `JWT_SECRET`.
- Do not expose the backend's database connection string or bootstrap administrator password in logs or issue reports.

## CI

Android and backend have separate GitHub Actions workflows. A green Android build alone does not establish that backend integration or live Neon connectivity works. See [implementation status](docs/IMPLEMENTATION_STATUS.md) for current limitations and outstanding integration work.
