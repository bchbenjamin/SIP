package com.sip.guardian.data.mapper;

import com.sip.guardian.data.remote.dto.IncidentDto;
import org.junit.Test;

import static org.junit.Assert.*;

public class IncidentMapperTest {

    @Test
    public void mapsFlatBackendIncidentFieldsIntoRoomEntity() {
        IncidentDto dto = new IncidentDto();
        dto.id = "incident-1";
        dto.state = "PENDING_VERIFICATION";
        dto.threatType = "WEAPON";
        dto.threatSeverity = "HIGH";
        dto.threatDescription = "Detected object";
        dto.latitude = 12.9716;
        dto.longitude = 77.5946;
        dto.locationReadable = "Bengaluru";
        dto.nodeId = "node-01";
        dto.createdAt = "2026-10-01T00:00:00Z";
        dto.updatedAt = "2026-10-01T00:01:00Z";

        IncidentMapper mapper = new IncidentMapper();
        var entity = mapper.toEntity(dto);

        assertEquals("incident-1", entity.id);
        assertEquals("PENDING_VERIFICATION", entity.state);
        assertEquals("WEAPON", entity.threatType);
        assertEquals("HIGH", entity.threatSeverity);
        assertEquals("Detected object", entity.threatDescription);
        assertEquals(12.9716, entity.latitude, 0.00001);
        assertEquals(77.5946, entity.longitude, 0.00001);
        assertEquals("Bengaluru", entity.locationReadable);
        assertEquals("node-01", entity.nodeId);
        assertTrue(entity.createdAtEpochMs > 0);
        assertTrue(entity.updatedAtEpochMs > entity.createdAtEpochMs);
    }

    @Test
    public void mapsLegacyNestedThreatAndLocationFields() {
        IncidentDto dto = new IncidentDto();
        dto.id = "incident-legacy";
        dto.state = "DETECTED";
        dto.threat = new IncidentDto.ThreatDto();
        dto.threat.type = "FIRE";
        dto.threat.severity = "MEDIUM";
        dto.threat.description = "Smoke";
        dto.location = new IncidentDto.LocationDto();
        dto.location.latitude = 1.5;
        dto.location.longitude = 2.5;
        dto.location.humanReadable = "Test location";

        var entity = new IncidentMapper().toEntity(dto);

        assertEquals("FIRE", entity.threatType);
        assertEquals("MEDIUM", entity.threatSeverity);
        assertEquals("Smoke", entity.threatDescription);
        assertEquals(1.5, entity.latitude, 0.00001);
        assertEquals(2.5, entity.longitude, 0.00001);
    }

    @Test
    public void preservesResponseEventsAcrossRoomCacheRoundTrip() {
        IncidentDto dto = new IncidentDto();
        dto.id = "incident-with-response";
        dto.state = "DETERRENCE_COMPLETED";
        dto.responseEvents = java.util.List.of(new IncidentDto.ResponseEventDto());
        dto.responseEvents.get(0).id = "response-1";
        dto.responseEvents.get(0).actionType = "DETERRENCE";
        dto.responseEvents.get(0).timestamp = "2026-10-01T00:00:00Z";
        dto.responseEvents.get(0).result = "completed";
        dto.responseEvents.get(0).autonomous = true;

        IncidentMapper mapper = new IncidentMapper();
        var entity = mapper.toEntity(dto);
        IncidentDto restored = mapper.toDto(entity);

        assertNotNull(entity.responseEventsJson);
        assertNotNull(restored.responseEvents);
        assertEquals(1, restored.responseEvents.size());
        assertEquals("response-1", restored.responseEvents.get(0).id);
        assertEquals("DETERRENCE", restored.responseEvents.get(0).actionType);
        assertTrue(restored.responseEvents.get(0).autonomous);
    }

}
