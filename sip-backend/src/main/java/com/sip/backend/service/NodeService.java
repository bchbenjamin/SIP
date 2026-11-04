package com.sip.backend.service;

import com.sip.backend.repository.NodeRepository;
import org.springframework.stereotype.Service;

@Service
public class NodeService {
    private final NodeRepository nodeRepository;

    public NodeService(NodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    // Stubs for CRUD and state methods
    public void updateNodeHeartbeat(String nodeId) {
        // Logic: update last_heartbeat timestamp
    }
    public String getNodeDetails(String nodeId) {
        return "Node details for " + nodeId;
    }
}