-- V1__initial_schema.sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'OPERATOR',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_role_check
        CHECK (role IN ('ADMIN', 'OPERATOR'))
);

CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'UNKNOWN',
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    last_heartbeat TIMESTAMPTZ,
    battery_level INTEGER,
    firmware_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT nodes_status_check
        CHECK (status IN ('ONLINE', 'DEGRADED', 'OFFLINE', 'UNKNOWN'))
);

CREATE TABLE incidents (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    threat_type TEXT NOT NULL,
    threat_severity TEXT NOT NULL,
    threat_description TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location_readable TEXT,
    location_accuracy DOUBLE PRECISION,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    autopilot_handled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
