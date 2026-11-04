-- V2__indexes.sql
CREATE INDEX idx_incidents_created_at
    ON incidents (created_at DESC);

CREATE INDEX idx_incidents_state
    ON incidents (state);

CREATE INDEX idx_incidents_node_id
    ON incidents (node_id);

CREATE INDEX idx_incidents_threat_type
    ON incidents (threat_type);

CREATE INDEX idx_incidents_updated_at
    ON incidents (updated_at DESC);
