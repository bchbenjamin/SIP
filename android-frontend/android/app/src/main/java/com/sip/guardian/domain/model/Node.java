package com.sip.guardian.domain.model;

import java.time.Duration;
import java.time.Instant;

/** A Raspberry Pi edge node in the mesh (plan §20 heartbeat logic). */
public final class Node {
    public static final Duration HEARTBEAT_INTERVAL = Duration.ofSeconds(30);
    public static final Duration DEGRADED_AFTER = Duration.ofSeconds(90);   // 1 missed cycle
    public static final Duration OFFLINE_AFTER = Duration.ofSeconds(90);    // 3+ missed cycles

    private final String nodeId;
    private final String name;
    private final Location location;
    private final NodeStatus status;
    private final Instant lastHeartbeat;
    private final Integer batteryLevel;
    private final String firmwareVersion;
    private final AutopilotPolicy autopilotPolicy;

    public Node(String nodeId, String name, Location location, NodeStatus status,
                Instant lastHeartbeat, Integer batteryLevel, String firmwareVersion,
                AutopilotPolicy autopilotPolicy) {
        this.nodeId = nodeId;
        this.name = name;
        this.location = location;
        this.status = status;
        this.lastHeartbeat = lastHeartbeat;
        this.batteryLevel = batteryLevel;
        this.firmwareVersion = firmwareVersion;
        this.autopilotPolicy = autopilotPolicy;
    }

    public String getNodeId() { return nodeId; }
    public String getName() { return name; }
    public Location getLocation() { return location; }
    public NodeStatus getStatus() { return status; }
    public Instant getLastHeartbeat() { return lastHeartbeat; }
    public Integer getBatteryLevel() { return batteryLevel; }
    public String getFirmwareVersion() { return firmwareVersion; }
    public AutopilotPolicy getAutopilotPolicy() { return autopilotPolicy; }

    public boolean isOnline() { return status == NodeStatus.ONLINE; }

    /** Heartbeat staleness — mirrors NodeService evaluation on the gateway. */
    public boolean isStale(Duration silence) {
        if (lastHeartbeat == null) return true;
        return Duration.between(lastHeartbeat, Instant.now()).compareTo(silence) > 0;
    }

    public Duration getTimeSinceHeartbeat() {
        if (lastHeartbeat == null) return Duration.ZERO;
        return Duration.between(lastHeartbeat, Instant.now());
    }
}
