package com.sip.backend.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * Allows Pi devices to authenticate via X-API-Key header.
 * Devices do NOT carry JWTs — they use a shared device API key.
 * This is acceptable for device-to-backend communication because:
 * - The key is stored server-side only (env var)
 * - Traffic should be TLS (Neon enforces SSL, Pi→backend should too)
 * - Device actions are logged via audit_events
 */
@Component
@Order(1)
public class ApiKeyAuthFilter extends OncePerRequestFilter {

    public static final String API_KEY_HEADER = "X-API-Key";

    @Value("${app.device-api-key:}")
    private String validApiKey;

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        String path = request.getRequestURI();

        // Only protect device-facing endpoints
        if (path.startsWith("/api/v1/incidents") ||
            path.startsWith("/api/v1/nodes/") && path.endsWith("/heartbeat")) {

            // Skip if already authenticated via JWT (Android app)
            if (request.getUserPrincipal() != null) {
                filterChain.doFilter(request, response);
                return;
            }

            // Check API key
            String presentedKey = request.getHeader(API_KEY_HEADER);
            if (validApiKey != null && !validApiKey.isBlank() &&
                MessageDigest.isEqual(validApiKey.getBytes(StandardCharsets.UTF_8),
                                      presentedKey.getBytes(StandardCharsets.UTF_8))) {
                filterChain.doFilter(request, response);
                return;
            }

            // API key missing or wrong
            if (presentedKey == null || presentedKey.isBlank()) {
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                response.setContentType("application/json");
                response.getWriter().write("{\"error\":\"Missing X-API-Key header\"}");
                return;
            }
            response.setStatus(HttpServletResponse.SC_FORBIDDEN);
            response.setContentType("application/json");
            response.getWriter().write("{\"error\":\"Invalid API key\"}");
            return;
        }

        filterChain.doFilter(request, response);
    }
}