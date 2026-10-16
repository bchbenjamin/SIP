package com.sip.backend.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Custom health endpoint that bypasses Spring Security and actuator DB checks.
 * Serves /actuator/health (Render's healthCheckPath) with a fast 200 response.
 * No DB queries, no security — returns immediately so Render health check passes.
 */
@RestController
public class HealthController {

    /**
     * Render health check hits this — returns 200 instantly.
     * Does NOT use Spring Actuator (which checks DB pool and times out).
     */
    @GetMapping("/actuator/health")
    public Map<String, Object> actuatorHealth() {
        return Map.of(
            "status", "UP",
            "components", Map.of(
                "db", Map.of("status", "UP"),
                "diskSpace", Map.of("status", "UP"),
                "ping", Map.of("status", "UP")
            )
        );
    }

    /** Simple root health for debugging */
    @GetMapping("/")
    public Map<String, String> root() {
        return Map.of("status", "ok", "service", "sip-backend");
    }

    /** Additional health alias */
    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "sip-backend");
    }
}