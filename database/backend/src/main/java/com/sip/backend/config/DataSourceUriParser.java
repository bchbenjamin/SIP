package com.sip.backend.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * Parses a Neon connection URI (e.g. postgresql://user:pass@host/db?sslmode=require)
 * and injects DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_QUERY into Spring's
 * Environment at context startup time — BEFORE property placeholders are resolved.
 *
 * This must be registered as a listener so it runs before the refresh starts.
 */
public class DataSourceUriParser {

    private static final Logger log = LoggerFactory.getLogger(DataSourceUriParser.class);

    private DataSourceUriParser() {}

    /**
     * Call this from your main method, before SpringApplication.run(...).
     * Sets system properties from DATABASE_URL if DB_HOST is not already set.
     */
    public static void parseAndSet(ConfigurableEnvironment env) {
        // Read env var directly — env.getProperty() may not yet see raw env vars
        // when this runs as an ApplicationContextInitializer (before property resolution).
        String uri = System.getenv("DATABASE_URL");

        // Fallback: if not set as env var, check if Spring already resolved it
        if (uri == null || uri.isBlank()) {
            uri = env.getProperty("DATABASE_URL", "");
        }

        if (uri == null || uri.isBlank() || uri.equals("jdbc:postgresql://localhost")) {
            log.warn("DATABASE_URL not set — datasource will use defaults");
            return;
        }

        log.info("DATABASE_URL parser initializing — DATABASE_URL='{}'",
                uri != null && uri.length() > 20 ? uri.substring(0, 20) + "..." : uri);

        if (System.getenv("DB_HOST") != null) {
            log.debug("DB_HOST already set in environment, skipping DATABASE_URL parse");
            return;
        }

        try {
            URI parsed = URI.create(uri);

            String userInfo = parsed.getRawUserInfo();
            String dbUser = "";
            String dbPassword = "";
            if (userInfo != null) {
                int colon = userInfo.indexOf(':');
                if (colon >= 0) {
                    dbUser = URLDecoder.decode(userInfo.substring(0, colon), StandardCharsets.UTF_8);
                    dbPassword = URLDecoder.decode(userInfo.substring(colon + 1), StandardCharsets.UTF_8);
                } else {
                    dbUser = URLDecoder.decode(userInfo, StandardCharsets.UTF_8);
                }
            }

            String hostPort = parsed.getHost();
            if (hostPort == null) {
                throw new IllegalArgumentException("No host in DATABASE_URL: " + uri);
            }
            if (parsed.getPort() != -1) {
                hostPort += ":" + parsed.getPort();
            }

            String dbName = parsed.getPath();
            if (dbName != null && dbName.startsWith("/")) {
                dbName = dbName.substring(1);
            }
            if (dbName == null || dbName.isBlank()) {
                dbName = "neondb";
            }

            String query = parsed.getRawQuery();
            if (query == null) query = "";
            if (!query.contains("sslmode")) {
                query = (query.isEmpty() ? "" : query + "&") + "sslmode=require";
            }

            Map<String, Object> props = new HashMap<>();
            props.put("DB_HOST", hostPort);
            props.put("DB_NAME", dbName);
            props.put("DB_USER", dbUser);
            props.put("DB_PASSWORD", dbPassword);
            props.put("DB_QUERY", query);

            // Remove any existing DB_HOST source so we can override
            env.getPropertySources().remove("DATA_SOURCE_URI_OVERRIDE");

            // Add highest-priority source before system properties
            env.getPropertySources()
                    .addFirst(new MapPropertySource("DATA_SOURCE_URI_OVERRIDE", props));

            log.info("Parsed DATABASE_URL → host={}, db={}, user={}",
                    hostPort, dbName, dbUser.isEmpty() ? "(none)" : dbUser);

        } catch (Exception e) {
            log.error("Failed to parse DATABASE_URL '{}': {}", uri, e.getMessage());
        }
    }
}