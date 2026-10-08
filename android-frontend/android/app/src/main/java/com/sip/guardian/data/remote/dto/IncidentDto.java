package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;
import java.util.List;

/** Accepts both the original Android wire model and the backend's canonical DTO shape. */
public class IncidentDto {
    public String id;
    public String state;

    // Canonical backend fields (the nested threat/location fields below remain supported).
    public String threatType;
    public String threatSeverity;
    public String threatDescription;
    public Double latitude;
    public Double longitude;
    public String locationReadable;
    public Double locationAccuracy;
    public String nodeName;

    public ThreatDto threat;
    public LocationDto location;
    public EvidenceDto evidence;
    public DetectionDto detection;
    public List<AnnotationDto> annotations;
    public List<ResponseEventDto> responseEvents;
    public DetailDto detail;
    public String nodeId;
    @SerializedName(value = "created_at", alternate = {"createdAt"}) public String createdAt;
    @SerializedName(value = "updated_at", alternate = {"updatedAt"}) public String updatedAt;
    public boolean autopilotHandled;

    public static class DetailDto {
        public List<EvidenceDto> evidence;
        public List<AnnotationDto> annotations;
        public List<ResponseEventDto> responseEvents;
    }

    public static class ThreatDto {
        public String type;
        public String severity;
        public String description;
    }

    public static class LocationDto {
        public double latitude;
        public double longitude;
        public String humanReadable;
        public double accuracy;
    }

    public static class EvidenceDto {
        public String id;
        public String imageUrl;
        public String videoUrl;
        public String audioUrl;
        public String thumbnailUrl;
        public String captureTimestamp;
        public String retentionExpiry;
        public boolean pendingUpload;
        public String type;
        public String storageKey;
        public String mimeType;
        public Long sizeBytes;
        public String sha256;
        public String uploadStatus;
    }

    public static class DetectionDto {
        public String predictedClass;
        public double confidence;
        public String modelVersion;
        public String detectionTimestamp;
        public List<String> sensorModalities;
    }

    public static class AnnotationDto {
        public String id;
        public String label;
        public String annotatorId;
        public String timestamp;
        public String notes;
        public Double confidence;
        public int version;
    }

    public static class ResponseEventDto {
        public String id;
        public String actionType;
        public String timestamp;
        public String result;
        public boolean autonomous;
    }
}
