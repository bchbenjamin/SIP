# SIP Guardian v0.2.0-beta.3

**Download:** `sip-guardian-v0.2.0-beta.3.apk`

---

## What's Fixed in Beta 3

- **Release Login Bug (R8 ClassCastException)**:
  - In previous release builds (`v0.2.0-beta.2` and earlier), successful 2xx authentication responses threw `ClassCastException` inside `AuthRepositoryImpl.login()`, resulting in the generic error *"Unexpected login failure"*.
  - Root cause: Previous R8 keep rules used `-keepclassmembers` for DTOs, preserving fields while allowing R8 to strip the `LoginResponse` class itself.
  - Solution: Verified and applied strict ProGuard rules preserving all DTO classes, domain models, and Retrofit API interfaces:
    ```proguard
    -keepattributes Signature, InnerClasses, EnclosingMethod, *Annotation*
    -keep class com.sip.guardian.domain.model.** { *; }
    -keep class com.sip.guardian.data.remote.dto.** { *; }
    -keep interface com.sip.guardian.data.remote.api.** { *; }
    ```
  - Bytecode Verification: Disassembly of `classes.dex` confirmed `com.sip.guardian.data.remote.dto.LoginResponse` survives with all required fields intact (`token`, `refreshToken`, `expiresIn`, `userId`, `username`, `role`).
  - Unit Tests: Added comprehensive `AuthRepositoryTest` validating 200 HTTP response deserialization, token persistence, 401/403 security exceptions, 500 internal server error handling, incomplete payloads, and malformed JSON.

---

## Verified Build Details

- **Package:** `com.sip.guardian`
- **Version:** `0.2.0-beta.3` (`versionCode 3`)
- **Min SDK:** 26 (Android 8.0)
- **Target SDK:** 35
- **Production API URL:** `https://api.sip.bchbenjamin.in/`
- **APK Size:** ~1.5 MB (R8 optimized, resource shrunk)
- **Signing Keystore:** `sip-beta.jks` (`CN=SIP Guardian, OU=Dev, O=SIP, L=City, ST=State, C=US`)
- **Signer SHA-256 Digest:** `5bba5bb3e83d1e1be8e9bb3173cf72e1d0bd3216f12513263580cfa931db6522`

---

## APK SHA-256 Checksum

```
99ab283e2d4f0c7f7774c05e8bb48fbc5b349313332c3dd660e258af20d609fe  sip-guardian-v0.2.0-beta.3.apk
```

To verify:
```bash
sha256sum sip-guardian-v0.2.0-beta.3.apk
```
