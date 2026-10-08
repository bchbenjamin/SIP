package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class AutopilotPolicyDto {
    public String nodeId;
    public boolean enabled;
    public List<String> allowedThreatTypes;
    public double confidenceThreshold;
    public int deterrenceTimeoutSeconds;
    public boolean autoEscalateOnDeterrenceFailure;
    public int maxDeterrenceAttempts;
    public List<String> alwaysEscalateTypes;
    public boolean requireMinimumConfidence;
    public boolean requireMultiModalConfirmation;
    public List<String> neverAutonomousTypes;
    public String syncState;
    @SerializedName(value = "lastSyncTimestamp", alternate = {"lastSync"})
    public String lastSyncTimestamp;
}
