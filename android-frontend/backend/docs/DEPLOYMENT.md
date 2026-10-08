# SIP Backend Deployment

## Local development

Requirements: JDK 17 and a working Gradle 8.9 wrapper.

Create a private environment file from the example, then export its values into the shell (Spring Boot does not automatically load `.env`):

```bash
cd backend
cp .env.example .env
set -a
source .env
set +a
./gradlew clean test
./gradlew bootRun
```

The API listens on `http://localhost:8080`; health is available at `/actuator/health`. Flyway migrations run on startup. Do not use real production secrets in a shared shell history or CI logs.

## Neon PostgreSQL

1. Create a Neon project and a development branch.
2. Copy the PostgreSQL connection details from Neon.
3. Set `DATABASE_URL` to a JDBC URL, not the Neon management API key. Ensure TLS is enabled.
4. URL-encode reserved characters in username/password values.

Example shape:

```text
DATABASE_URL=jdbc:postgresql://HOST/DATABASE?user=USERNAME&password=URL_ENCODED_PASSWORD&sslmode=require
```

Keep this value private. Test migrations on a disposable Neon branch before applying them to a database containing important data.

## First administrator

There are no default login credentials. Before the first backend startup, optionally set both `SIP_BOOTSTRAP_ADMIN_USERNAME` and `SIP_BOOTSTRAP_ADMIN_PASSWORD`; the password must be at least 12 characters. After the administrator is created, remove these environment variables. The bootstrap will not change an existing user's role or password.

## Deployment checklist

- Use HTTPS and WSS.
- Configure a strong unique `JWT_SECRET` (at least 32 bytes).
- Configure `DATABASE_URL` privately; do not put it in Android or commit it.
- Restrict network access to the backend and database where possible.
- Configure health checks and logs without secrets.
- Verify Flyway migration state and backup/restore procedures.
- Do not advertise evidence media as available until an object-storage provider is configured.
- Do not consider Raspberry Pi integration complete until node authentication, MQTT ingestion, heartbeat handling, deduplication and end-to-end tests are implemented.

## CI

The backend workflow runs `gradle clean test bootJar` on JDK 17. Locally, use `./gradlew clean test bootJar`.
