package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;

public class NodeDto {
    @SerializedName(value = "nodeId", alternate = {"id"})
    public String nodeId;
    public String name;
    public String status;
    public double latitude;
    public double longitude;
    public String lastHeartbeat;
    public Integer batteryLevel;
    public String firmwareVersion;
    public AutopilotPolicyDto autopilotPolicy;
}
