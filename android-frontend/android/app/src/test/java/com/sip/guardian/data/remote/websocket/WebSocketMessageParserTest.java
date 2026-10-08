package com.sip.guardian.data.remote.websocket;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import org.junit.Test;

public class WebSocketMessageParserTest {
    private final WebSocketMessageParser parser = new WebSocketMessageParser();

    @Test public void parsesIncidentEvent() {
        WebSocketEvent event = parser.parse(
                "{\"type\":\"INCIDENT_NEW\",\"payload\":{\"id\":\"i-1\",\"state\":\"DETECTED\",\"threatType\":\"WEAPON\"}}");
        assertNotNull(event);
        assertNotNull(((WebSocketEvent.OnIncidentReceived) event).incident);
        assertEquals("WEAPON", ((WebSocketEvent.OnIncidentReceived) event).incident.threatType);
    }

    @Test public void acceptsBackendIncidentUpdatedPayload() {
        WebSocketEvent event = parser.parse(
                "{\"type\":\"INCIDENT_UPDATED\",\"payload\":{\"id\":\"i-2\",\"state\":\"VERIFIED\"}}");
        assertNotNull(event);
        WebSocketEvent.OnIncidentUpdated updated = (WebSocketEvent.OnIncidentUpdated) event;
        assertEquals("i-2", updated.incidentId);
        assertEquals("VERIFIED", updated.state);
        assertNotNull(updated.incident);
        assertEquals("i-2", updated.incident.id);
        assertEquals("VERIFIED", updated.incident.state);
    }

    @Test public void acceptsBackendNodeStatusEventName() {
        WebSocketEvent event = parser.parse(
                "{\"type\":\"NODE_STATUS_CHANGED\",\"payload\":{\"id\":\"node-1\",\"status\":\"ONLINE\"}}");
        assertNotNull(event);
        WebSocketEvent.OnNodeStatusChanged status = (WebSocketEvent.OnNodeStatusChanged) event;
        assertEquals("node-1", status.nodeId);
        assertEquals("ONLINE", status.status);
    }

    @Test public void dropsMalformedMessages() {
        assertNull(parser.parse("{not-json"));
        assertNull(parser.parse("{\"type\":\"UNKNOWN\"}"));
    }
}
