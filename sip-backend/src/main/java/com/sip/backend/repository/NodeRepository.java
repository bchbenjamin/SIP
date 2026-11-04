package com.sip.backend.repository;

import com.sip.backend.domain.Node;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface NodeRepository extends JpaRepository<Node, String> {
    Optional<Node> findById(String nodeId);
}