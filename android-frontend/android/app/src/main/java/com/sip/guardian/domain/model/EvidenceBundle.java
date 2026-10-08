package com.sip.guardian.domain.model;

import java.time.Instant;
import java.util.EnumSet;
import java.util.Set;

/**
 * Groups all evidence references for an incident.
 * URLs are authenticated gateway endpoints; binary is never inline.
 */
public final class EvidenceBundle {
    private final String imageUrl;
    private final String videoUrl;
    private final String audioUrl;
    private final String thumbnailUrl;
    private final Instant captureTimestamp;
    private final Instant retentionExpiry;
    private final boolean pendingUpload; // evidence_pending flag (plan §27)

    public EvidenceBundle(String imageUrl, String videoUrl, String audioUrl,
                          String thumbnailUrl, Instant captureTimestamp,
                          Instant retentionExpiry, boolean pendingUpload) {
        this.imageUrl = imageUrl;
        this.videoUrl = videoUrl;
        this.audioUrl = audioUrl;
        this.thumbnailUrl = thumbnailUrl;
        this.captureTimestamp = captureTimestamp;
        this.retentionExpiry = retentionExpiry;
        this.pendingUpload = pendingUpload;
    }

    public String getImageUrl() { return imageUrl; }
    public String getVideoUrl() { return videoUrl; }
    public String getAudioUrl() { return audioUrl; }
    public String getThumbnailUrl() { return thumbnailUrl; }
    public Instant getCaptureTimestamp() { return captureTimestamp; }
    public Instant getRetentionExpiry() { return retentionExpiry; }
    public boolean isPendingUpload() { return pendingUpload; }

    public boolean hasImage() { return imageUrl != null && !imageUrl.isEmpty(); }
    public boolean hasVideo() { return videoUrl != null && !videoUrl.isEmpty(); }
    public boolean hasAudio() { return audioUrl != null && !audioUrl.isEmpty(); }

    public Set<EvidenceType> getAvailableTypes() {
        Set<EvidenceType> types = EnumSet.noneOf(EvidenceType.class);
        if (hasImage()) types.add(EvidenceType.IMAGE);
        if (hasVideo()) types.add(EvidenceType.VIDEO);
        if (hasAudio()) types.add(EvidenceType.AUDIO);
        return types;
    }
}
