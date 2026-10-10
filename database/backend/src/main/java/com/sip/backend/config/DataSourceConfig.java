package com.sip.backend.config;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;

/**
 * Parses a Neon connection URI (e.g. postgresql://user:pass@host/db?sslmode=require)
 * and sets individual environment variables consumed by application.yml datasource.
 *
 * Fallback: if DB_HOST is already set, does nothing.
 */
@Configuration
public class DataSourceConfig {

    private static final Logger log = LoggerFactory.getLogger(DataSourceConfig.class);

    @PostConstruct
    public void parseNeonUri() {
        String uri = System.getenv("DATABASE_URL");
        if (uri == null || uri.isBlank()) {
            log.warn("DATABASE_URL is not set — datasource will use default values");
            return;
        }
        if (System.getenv("DB_HOST") != null) {
            log.debug("DB_HOST already set, skipping DATABASE_URL parse");
            return;
        }

        try {
            URI parsed = URI.create(uri);

            String userInfo = parsed.getRawUserInfo();
            String dbUser = "";
            String dbPassword = "";
            if (userInfo != null) {
                int colon = userInfo.indexOf(':');
                dbUser = colon >= 0
                        ? URLDecoder.decode(userInfo.substring(0, colon), StandardCharsets.UTF_8)
                        : URLDecoder.decode(userInfo, StandardCharsets.UTF_8);
                dbPassword = colon >= 0
                        ? URLDecoder.decode(userInfo.substring(colon + 1), StandardCharsets.UTF_8)
                        : "";
            }

            String hostPort = parsed.getHost();
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
            // Append sslmode=require if not present
            if (!query.contains("sslmode")) {
                query = (query.isEmpty() ? "" : query + "&") + "sslmode=require";
            }

            setEnv("DB_HOST", hostPort);
            setEnv("DB_NAME", dbName);
            setEnv("DB_USER", dbUser);
            setEnv("DB_PASSWORD", dbPassword);
            setEnv("DB_QUERY", query);

            log.info("Parsed DATABASE_URL → host={}, db={}, user={}",
                    hostPort, dbName,
                    dbUser.isEmpty() ? "(none)" : dbUser);
        } catch (Exception e) {
            log.error("Failed to parse DATABASE_URL '{}': {}", uri, e.getMessage());
        }
    }

    private void setEnv(String key, String value) {
        if (System.getenv(key) == null) {
            System.setProperty(key, value);
        }
    }
}