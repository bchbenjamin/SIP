package com.sip.backend.controller;

import com.sip.backend.dto.NodeHeartbeatPayload;
import com.sip.backend.entity.Node;
import com.sip.backend.entity.NodeHeartbeat;
import com.sip.backend.repository.NodeRepository;
import com.sip.backend.service.NodeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/nodes")
public class NodeController {

    private final NodeService nodeService;
    private final NodeRepository nodeRepository;

    public NodeController(NodeService nodeService, NodeRepository nodeRepository) {
        this.nodeService = nodeService;
        this.nodeRepository = nodeRepository;
    }

    @GetMapping
    public ResponseEntity<?> listNodes() {
        return ResponseEntity.ok(nodeRepository.findAll());
    }

    @GetMapping("/{id}")
    public ResponseEntity<?> getNode(@PathVariable String id) {
        return nodeService.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<?> registerNode(@RequestBody Node node) {
        if (node.id == null || node.id.isBlank()) {
            return ResponseEntity.badRequest().build();
        }
        if (nodeRepository.existsById(node.id)) {
            return ResponseEntity.status(409).body("Node already registered");
        }
        Node saved = nodeRepository.save(node);
        return ResponseEntity.status(201).body(saved);
    }

    @PostMapping("/{id}/heartbeat")
    public ResponseEntity<?> heartbeat(@PathVariable String id, @RequestBody NodeHeartbeatPayload payload) {
        try {
            payload.setNodeId(id);
            NodeHeartbeat hb = nodeService.recordHeartbeat(id, payload);
            return ResponseEntity.ok(hb);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @PutMapping("/{id}/status")
    public ResponseEntity<?> updateStatus(@PathVariable String id, @RequestParam String status) {
        return nodeService.findById(id).map(node -> {
            try {
                node.status = Node.NodeStatus.valueOf(status.toUpperCase());
                nodeRepository.save(node);
                return ResponseEntity.ok(node);
            } catch (IllegalArgumentException e) {
                return ResponseEntity.badRequest();
            }
        }).orElse(ResponseEntity.notFound().build());
    }
}