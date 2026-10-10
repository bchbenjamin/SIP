# SIP Guardian v0.2.0 Beta

**Download:** `app-release.apk` (SHA-256 below)

---

## What's New

- **Edge-AI threat detection** — YOLO/SSD inference on Raspberry Pi 5
- **Autonomous deterrence** — GPIO-triggered sirens/strobes for deterrable threats
- **Real-time operator app** — Android app with live incident feed via WebSocket
- **MQTT + REST fallback** — incidents published over MQTT (QoS 1) with SQLite offline queue
- **PostgreSQL backend** — Spring Boot + Neon, Flyway migrations, JWT auth

## Installation

1. Enable "Install from unknown sources" in Android settings
2. Transfer `app-release.apk` to your device
3. Open and install

## API Configuration

The APK is configured to connect to:

```
https://api.sip.bchbenjamin.in/
```

**Do not use** this beta for real emergency response. This is a research prototype.

## Verified Build

- **Package:** `com.sip.guardian`
- **Version:** 0.2.0 (versionCode 2)
- **Min SDK:** 26 (Android 8.0)
- **Target SDK:** 35
- **API URL:** `https://api.sip.bchbenjamin.in/`
- **APK Size:** 1.5 MB
- **Build:** Release (R8 minified, signed)
- **Signature:** Self-signed beta key (CN=SIP Guardian Beta)

## APK SHA-256

```
86c77e25f17e09f30e5f6bd5812e05890b7e1c7fae37cef5677858815c75b50b
```

Verify with:
```bash
sha256sum app-release.apk
# or
Get-FileHash app-release.apk -Algorithm SHA256
```

## Known Limitations

- Detection accuracy depends on camera angle, lighting, and model weights
- No integration with real police or emergency services
- Beta software — expect bugs, false positives, and missed detections
- Do not use as a substitute for professional security or emergency response
- WebSocket reconnect may need manual app restart after backend sleep (free Render tier)

## Repository

https://github.com/bchbenjamin/SIP