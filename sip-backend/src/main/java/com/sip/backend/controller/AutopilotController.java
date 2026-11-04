package com.sip.backend.controller;

import com.sip.backend.service.AutopilotService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/autopilot")
public class AutopilotController {

    private final AutopilotService autopilotService;

    public AutopilotController(AutopilotService autopilotService) {
        this.autopilotService = autopilotService;
    }

    @GetMapping("/policy")
    public ResponseEntity<Map<String, Object>> getPolicy(@RequestParam String nodeId) {
        // TODO: Fetch policy using NodeService/AutopilotService and return contract shape
        return ResponseEntity.ok(Map.of("policy", "policy_data_stub"));
    }

    @PutMapping("/policy")
    public ResponseEntity<?> updatePolicy(@RequestParam String nodeId, @RequestBody Map<String, Object> policyData) {
        // TODO: Validate policy data before saving (Increment C/D validation)
        // autopilotService.updatePolicy(nodeId, newPolicy);
        return ResponseEntity.ok(Map.of("status", "Policy update accepted"));
    }
}