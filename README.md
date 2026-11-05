# SIP Guardian — AI-Powered Street Safety Device Network

[![Android CI](https://github.com/bchbenjamin/SIP/actions/workflows/android-ci.yml/badge.svg?branch=main)](https://github.com/bchbenjamin/SIP/actions/workflows/android-ci.yml)
[![Backend CI](https://github.com/bchbenjamin/SIP/actions/workflows/backend-ci.yml/badge.svg?branch=main)](https://github.com/bchbenjamin/SIP/actions/workflows/backend-ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/bchbenjamin/SIP?color=blue)](https://github.com/bchbenjamin/SIP/releases/tag/v0.3.0)

Autonomous, multi-tier threat detection and street safety network connecting edge compute cores (Raspberry Pi 5) with a canonical cloud backend (Spring Boot 3 / Neon PostgreSQL) and real-time operator client (Android).

---

## Current Status (November 5, 2026)

* **Production Backend:** **LIVE & OPERATIONAL** at `https://api.sip.bchbenjamin.in/`
* **Latest Android Release:** [`v0.3.0`](https://github.com/bchbenjamin/SIP/releases/tag/v0.3.0) (`sip-guardian-v0.3.0.apk`)
* **Verification Audit:** Complete forensic audit documented in [`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md) and [`android-frontend/docs/IMPLEMENTATION_STATUS.md`](android-frontend/docs/IMPLEMENTATION_STATUS.md).

---

## System Architecture

```text
 Raspberry Pi 5 (Edge Compute Core)
 [YOLOv8 Edge Inference | Autonomous Offline Response | MQTT / REST Sync]
                             │
                             ▼ (HTTPS / WSS / MQTT)
 Canonical Cloud Backend (database/backend)
 [Spring Boot 3 | Java 17 | Render Web Service | Neon PostgreSQL]
                             ▲
                             │ (HTTPS / WSS)
 Android Client (android-frontend/android)
 [Kotlin / Jetpack Compose | Java Domain & Data | Room Local Cache]
```

### Component Directory Layout

- **`android-frontend/android/`** — Android application (Jetpack Compose UI, Room offline cache, Retrofit REST client, WebSocket foreground service).
- **`database/backend/`** — Canonical Spring Boot 3 backend service (Neon PostgreSQL persistence, Flyway migrations, JWT security, WebSocket broadcaster, incident state machine).
- **`node-backend/pi/`** — Raspberry Pi 5 edge compute scripts (YOLOv8 threat detection, edge WebSocket server, autonomous training pipeline, undervoltage monitoring).
- **`landing-page/`** — TanStack Start SSR frontend deployed on Vercel at `sip.bchbenjamin.in`.
- **`docs/`** — Architectural specifications, verification audit reports, and implementation status.

---

## Latest Releases

### Android Application (`v0.3.0`)
- **APK Asset:** [`sip-guardian-v0.3.0.apk`](https://github.com/bchbenjamin/SIP/releases/download/v0.3.0/sip-guardian-v0.3.0.apk)
- **SHA-256 Checksum:** `11e0d5893be64f183e9f7dcf8163647ec467de6870b21c69364b6d1412bd7cc4`
- **Signing:** APK Signature Scheme v2 (Signed with `sip-beta.jks`)
- **Target SDK:** Android 15 (API 35), Min SDK: Android 8.0 (API 26)

### Production Backend API
- **Base URL:** `https://api.sip.bchbenjamin.in/`
- **Health Check:** `https://api.sip.bchbenjamin.in/health` → `{"status":"UP"}`
- **Actuator Endpoint:** `https://api.sip.bchbenjamin.in/actuator/health` → `{"service":"sip-backend","status":"UP"}`

---

## Local Development & Build

### Prerequisites
- JDK 17 (Eclipse Temurin recommended)
- Gradle 8.9+
- Android SDK (API 35, Build-Tools 35.0.0)
- Python 3.11+ (for edge ML pipelines)

### 1. Build Canonical Backend
```bash
cd database/backend
./gradlew clean test bootJar
```

### 2. Build Android App
```bash
cd android-frontend/android
./gradlew clean test assembleRelease
```

---

## License & Attribution

Developed as part of the **AI-Powered Street Safety Device Network** (ASIP 24UTAI13) at Atria Institute of Technology, Bengaluru.
All rights reserved.
