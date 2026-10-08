package com.sip.backend.service;

import com.sip.backend.dto.NodeHeartbeatPayload;
import com.sip.backend.entity.Node;
import com.sip.backend.entity.NodeHeartbeat;
import com.sip.backend.repository.NodeHeartbeatRepository;
import com.sip.backend.repository.NodeRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

@Service
public class NodeService {

    private static final Logger log = LoggerFactory.getLogger(NodeService.class);

    private final NodeRepository nodeRepository;
    private final NodeHeartbeatRepository heartbeatRepository;

    public NodeService(NodeRepository nodeRepository, NodeHeartbeatRepository heartbeatRepository) {
        this.nodeRepository = nodeRepository;
        this.heartbeatRepository = heartbeatRepository;
    }

    @Transactional
    public NodeHeartbeat recordHeartbeat(String nodeId, NodeHeartbeatPayload payload) {
        Node node = nodeRepository.findById(nodeId)
                .orElseThrow(() -> new IllegalArgumentException("Unknown node: " + nodeId));

        // Update node status and last heartbeat
        node.setLastHeartbeat(payload.getTimestamp() != null ? payload.getTimestamp() : Instant.now());
        node.setStatus(Node.NodeStatus.ONLINE);
        if (payload.getBatteryLevel() != null) node.setBatteryLevel(payload.getBatteryLevel());
        nodeRepository.save(node);

        // Record heartbeat
        NodeHeartbeat heartbeat = new NodeHeartbeat();
        heartbeat.setId(UUID.randomUUID().toString());
        heartbeat.setNodeId(nodeId);
        heartbeat.setTimestamp(payload.getTimestamp() != null ? payload.getTimestamp() : Instant.now());
        heartbeat.setBatteryLevel(payload.getBatteryLevel());
        heartbeat.setTemperature(payload.getTemperature());
        heartbeat.setStorageAvailableMb(payload.getStorageAvailableMb());
        heartbeat.setModelVersion(payload.getModelVersion());
        heartbeat.setQueueDepth(payload.getQueueDepth() != null ? payload.getQueueDepth() : 0);

        heartbeat = heartbeatRepository.save(heartbeat);
        log.debug("Heartbeat recorded for node {} (queue_depth={})", nodeId, heartbeat.getQueueDepth());
        return heartbeat;
    }

    @Transactional
    public void markNodeOffline(String nodeId) {
        nodeRepository.findById(nodeId).ifPresent(node -> {
            node.setStatus(Node.NodeStatus.OFFLINE);
            nodeRepository.save(node);
            log.info("Node {} marked OFFLINE", nodeId);
        });
    }

    public Optional<Node> findById(String nodeId) {
        return nodeRepository.findById(nodeId);
    }
}