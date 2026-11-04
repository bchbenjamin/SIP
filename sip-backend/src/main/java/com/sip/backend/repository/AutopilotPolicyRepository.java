package com.sip.backend.repository;

import com.sip.backend.domain.AutopilotPolicy;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface AutopilotPolicyRepository extends JpaRepository<AutopilotPolicy, String> {
    Optional<AutopilotPolicy> findByNodeId(String nodeId);
}