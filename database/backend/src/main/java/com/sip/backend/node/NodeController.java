package com.sip.backend.node;

import com.sip.backend.dto.NodeDto;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/v1/nodes")
public class NodeController {

    private final NodeService nodeService;

    public NodeController(NodeService nodeService) {
        this.nodeService = nodeService;
    }

    @GetMapping
    public ResponseEntity<List<NodeDto>> getAllNodes() {
        return ResponseEntity.ok(nodeService.getAllNodes());
    }

    @GetMapping("/{id}")
    public ResponseEntity<NodeDto> getNode(@PathVariable String id) {
        return ResponseEntity.ok(nodeService.getNode(id));
    }
}