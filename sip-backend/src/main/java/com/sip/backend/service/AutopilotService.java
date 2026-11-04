package com.sip.backend.service;

import com.sip.backend.repository.AutopilotPolicyRepository;
import com.sip.backend.domain.AutopilotPolicy;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class AutopilotService {
    private final AutopilotPolicyRepository policyRepository;

    public AutopilotService(AutopilotPolicyRepository policyRepository) {
        this.policyRepository = policyRepository;
    }

    public Optional<AutopilotPolicy> getPolicyForNode(String nodeId) {
        return policyRepository.findByNodeId(nodeId);
    }

    public void updatePolicy(String nodeId, AutopilotPolicy policy) {
        // Logic: save/update policy, trigger audit event
    }
}