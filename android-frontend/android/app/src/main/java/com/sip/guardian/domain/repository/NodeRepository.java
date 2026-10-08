package com.sip.guardian.domain.repository;

import com.sip.guardian.domain.model.Node;

import java.util.List;

public interface NodeRepository {
    List<Node> getNodes();
    Node getNodeById(String nodeId);
}
