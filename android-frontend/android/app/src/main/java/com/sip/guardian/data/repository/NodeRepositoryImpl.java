package com.sip.guardian.data.repository;

import com.google.gson.Gson;
import com.sip.guardian.data.local.dao.NodeDao;
import com.sip.guardian.data.local.entity.NodeEntity;
import com.sip.guardian.data.remote.api.SipApiService;
import com.sip.guardian.data.remote.dto.NodeDto;
import com.sip.guardian.domain.model.AutopilotPolicy;
import com.sip.guardian.domain.model.Location;
import com.sip.guardian.domain.model.Node;
import com.sip.guardian.domain.model.NodeStatus;
import com.sip.guardian.domain.repository.NodeRepository;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import javax.inject.Inject;
import javax.inject.Singleton;

@Singleton
public class NodeRepositoryImpl implements NodeRepository {

    private final SipApiService api;
    private final NodeDao dao;
    private final Gson gson = new Gson();

    @Inject
    public NodeRepositoryImpl(SipApiService api, NodeDao dao) {
        this.api = api;
        this.dao = dao;
    }

    @Override
    public List<Node> getNodes() {
        try {
            var response = api.getNodes().execute();
            if (response.isSuccessful() && response.body() != null) {
                List<NodeEntity> entities = new ArrayList<>();
                List<Node> out = new ArrayList<>();
                for (NodeDto dto : response.body()) {
                    if (dto == null || dto.nodeId == null || dto.nodeId.isBlank()) continue;
                    entities.add(toEntity(dto));
                    out.add(toDomain(dto));
                }
                dao.upsertAll(entities);
                return out;
            }
        } catch (IOException ignored) {
            // Use the last known cache when the backend cannot be reached.
        } catch (RuntimeException ignored) {
            // Malformed individual API data should not crash the dashboard.
        }

        List<Node> cached = new ArrayList<>();
        for (NodeEntity e : dao.getAll()) cached.add(toDomain(e));
        return cached;
    }

    @Override
    public Node getNodeById(String nodeId) {
        if (nodeId == null) return null;
        for (Node n : getNodes()) {
            if (nodeId.equals(n.getNodeId())) return n;
        }
        return null;
    }

    private NodeEntity toEntity(NodeDto dto) {
        NodeEntity e = new NodeEntity();
        e.nodeId = dto.nodeId;
        e.name = dto.name;
        e.status = dto.status;
        e.latitude = dto.latitude;
        e.longitude = dto.longitude;
        e.lastHeartbeatEpochMs = parseInstant(dto.lastHeartbeat) != null
                ? parseInstant(dto.lastHeartbeat).toEpochMilli() : 0;
        e.batteryLevel = dto.batteryLevel;
        e.firmwareVersion = dto.firmwareVersion;
        e.policyJson = dto.autopilotPolicy != null ? gson.toJson(dto.autopilotPolicy) : null;
        return e;
    }

    private Node toDomain(NodeDto dto) {
        return new Node(dto.nodeId, dto.name,
                new Location(dto.latitude, dto.longitude, "", 0),
                safeStatus(dto.status), parseInstant(dto.lastHeartbeat),
                dto.batteryLevel, dto.firmwareVersion, AutopilotPolicy.disabled());
    }

    private static Instant parseInstant(String value) {
        if (value == null || value.isBlank()) return null;
        try { return Instant.parse(value); }
        catch (RuntimeException ignored) {
            try { return java.time.OffsetDateTime.parse(value).toInstant(); }
            catch (RuntimeException ignoredAgain) { return null; }
        }
    }

    private static NodeStatus safeStatus(String value) {
        if (value == null) return NodeStatus.UNKNOWN;
        try { return NodeStatus.valueOf(value); }
        catch (IllegalArgumentException ignored) { return NodeStatus.UNKNOWN; }
    }

    private Node toDomain(NodeEntity e) {
        return new Node(e.nodeId, e.name,
                new Location(e.latitude, e.longitude, "", 0), safeStatus(e.status),
                e.lastHeartbeatEpochMs > 0 ? Instant.ofEpochMilli(e.lastHeartbeatEpochMs) : null,
                e.batteryLevel, e.firmwareVersion, AutopilotPolicy.disabled());
    }
}
