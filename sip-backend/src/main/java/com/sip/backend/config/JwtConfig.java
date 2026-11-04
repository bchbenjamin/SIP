package com.sip.backend.config;

public class JwtConfig {
    // These values must come from application.yml/environment variables
    public static final String SECRET = "A_VERY_LONG_AND_COMPLEX_SECRET_KEY_MUST_BE_LOADED_FROM_ENV";
    public static final long EXPIRATION_MS = 900_000; // 15 minutes (900 seconds)
    public static final long REFRESH_EXPIRATION_MS = 604_800_000; // 7 days (long-lived, revocable)
}