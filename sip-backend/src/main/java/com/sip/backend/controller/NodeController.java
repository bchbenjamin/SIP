package com.sip.backend.controller;

import com.sip.backend.service.NodeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/nodes")
public class NodeController {

    private final NodeService nodeService;

    public NodeController(NodeService nodeService) {
        this.nodeService = nodeService;
    }

    @GetMapping
    public ResponseEntity<Map<String, Object>> getAllNodes() {
        // TODO: Implement node listing from DB/Service
        return ResponseEntity.ok(Map.of("nodes", List.of()));
    }

    @GetMapping("/{id}")
    public ResponseEntity<String> getNodeDetails(@PathVariable String id) {
        return ResponseEntity.ok(nodeService.getNodeDetails(id));
    }
}