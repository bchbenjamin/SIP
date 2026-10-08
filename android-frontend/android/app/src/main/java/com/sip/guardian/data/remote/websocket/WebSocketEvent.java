package com.sip.guardian.data.remote.websocket;

import com.sip.guardian.data.remote.dto.IncidentDto;

/** Typed events emitted by SipWebSocketClient (Observer pattern, plan §10). */
public abstract class WebSocketEvent {

    public enum ConnectionState { CONNECTING, CONNECTED, DISCONNECTED, FAILED }

    public static final class OnConnectionStateChanged extends WebSocketEvent {
        public final ConnectionState state;
        public final String reason;
        public OnConnectionStateChanged(ConnectionState state, String reason) {
            this.state = state; this.reason = reason;
        }
    }

    public static final class OnIncidentReceived extends WebSocketEvent {
        public final IncidentDto incident;
        public OnIncidentReceived(IncidentDto incident) { this.incident = incident; }
    }

    public static final class OnIncidentUpdated extends WebSocketEvent {
        public final String incidentId;
        public final String state;
        /** Full DTO when supplied by the backend; null for legacy state-only events. */
        public final IncidentDto incident;
        public OnIncidentUpdated(String incidentId, String state) {
            this.incidentId = incidentId;
            this.state = state;
            this.incident = null;
        }
        public OnIncidentUpdated(IncidentDto incident) {
            this.incident = incident;
            this.incidentId = incident != null ? incident.id : null;
            this.state = incident != null ? incident.state : null;
        }
    }

    public static final class OnNodeStatusChanged extends WebSocketEvent {
        public final String nodeId;
        public final String status;
        public final String lastHeartbeat;
        public OnNodeStatusChanged(String nodeId, String status, String lastHeartbeat) {
            this.nodeId = nodeId; this.status = status; this.lastHeartbeat = lastHeartbeat;
        }
    }

    public static final class OnPolicySynced extends WebSocketEvent {
        public final String nodeId;
        public final String syncState;
        public OnPolicySynced(String nodeId, String syncState) {
            this.nodeId = nodeId; this.syncState = syncState;
        }
    }

    public static final class OnSystemAlert extends WebSocketEvent {
        public final String level;
        public final String message;
        public final String nodeId;
        public OnSystemAlert(String level, String message, String nodeId) {
            this.level = level; this.message = message; this.nodeId = nodeId;
        }
    }
}
