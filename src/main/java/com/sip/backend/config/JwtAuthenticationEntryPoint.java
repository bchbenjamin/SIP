package com.sip.backend.config;

import org.springframework.security.core.AuthenticationException;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.http.ResponseEntity;
import java.time.ZonedDateTime;
import java.util.Map;

@RestControllerAdvice
public class JwtAuthenticationEntryPoint {

    @ExceptionHandler(AuthenticationException.class)
    public ResponseEntity<Map<String, Object>> handleAuthenticationException(AuthenticationException ex) {
        // Map this to AUTH_INVALID_CREDENTIALS or AUTH_SESSION_EXPIRED
        return ResponseEntity.status(401).body(
            Map.of("timestamp", ZonedDateTime.now().toString(),
                   "status", 401,
                   "code", "AUTH_INVALID_CREDENTIALS",
                   "message", "Authentication required or token invalid",
                   "path", "/api/v1/auth/login",
                   "requestId", "..."
            )
        );
    }
}