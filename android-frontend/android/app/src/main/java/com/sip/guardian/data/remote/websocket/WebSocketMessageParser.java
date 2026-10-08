package com.sip.guardian.data.remote.websocket;

import javax.inject.Inject;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sip.guardian.data.remote.dto.IncidentDto;

/** Parses backend events and safely drops malformed or unknown messages. */
public class WebSocketMessageParser {
    private final Gson gson;

    @Inject
    public WebSocketMessageParser() {
        this.gson = new Gson();
    }

    public WebSocketEvent parse(String text) {
        try {
            JsonObject root = JsonParser.parseString(text).getAsJsonObject();
            if (!root.has("type") || !root.get("type").isJsonPrimitive()) return null;
            String type = root.get("type").getAsString();
            JsonObject payload = root.has("payload") && root.get("payload").isJsonObject()
                    ? root.getAsJsonObject("payload") : new JsonObject();

            switch (type) {
                case "INCIDENT_NEW":
                    return new WebSocketEvent.OnIncidentReceived(gson.fromJson(payload, IncidentDto.class));
                case "INCIDENT_UPDATED": {
                    IncidentDto incident = gson.fromJson(payload, IncidentDto.class);
                    if (incident != null && incident.id != null) {
                        return new WebSocketEvent.OnIncidentUpdated(incident);
                    }
                    return new WebSocketEvent.OnIncidentUpdated(
                            first(payload, "incidentId", "id"), str(payload, "state"));
                }
                case "NODE_STATUS":
                case "NODE_STATUS_CHANGED":
                    return new WebSocketEvent.OnNodeStatusChanged(
                            first(payload, "nodeId", "id"), str(payload, "status"),
                            first(payload, "lastHeartbeat", "last_heartbeat"));
                case "POLICY_SYNCED":
                    return new WebSocketEvent.OnPolicySynced(
                            str(payload, "nodeId"), str(payload, "syncState"));
                case "SYSTEM_ALERT":
                    return new WebSocketEvent.OnSystemAlert(
                            str(payload, "level"), str(payload, "message"), str(payload, "nodeId"));
                default:
                    return null;
            }
        } catch (Exception e) {
            return null;
        }
    }

    private static String first(JsonObject o, String first, String second) {
        String value = str(o, first);
        return value != null ? value : str(o, second);
    }

    private static String str(JsonObject o, String key) {
        return o.has(key) && !o.get(key).isJsonNull() ? o.get(key).getAsString() : null;
    }
}
