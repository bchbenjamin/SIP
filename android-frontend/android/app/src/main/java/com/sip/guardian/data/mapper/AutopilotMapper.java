package com.sip.guardian.data.mapper;

import com.sip.guardian.data.remote.dto.AutopilotPolicyDto;
import com.sip.guardian.domain.model.AutopilotPolicy;
import com.sip.guardian.domain.model.AutopilotState;
import com.sip.guardian.domain.model.EscalationRules;
import com.sip.guardian.domain.model.SafetyConstraints;
import com.sip.guardian.domain.model.ThreatType;

import java.time.Instant;
import java.util.EnumSet;
import java.util.Set;
import java.util.stream.Collectors;

public class AutopilotMapper {

    public AutopilotPolicy toDomain(AutopilotPolicyDto dto) {
        if (dto == null) return AutopilotPolicy.disabled();

        Set<ThreatType> allowed = parseThreatTypes(dto.allowedThreatTypes);
        Set<ThreatType> alwaysEscalate = parseThreatTypes(dto.alwaysEscalateTypes);
        Set<ThreatType> neverAutonomous = parseThreatTypes(dto.neverAutonomousTypes);

        return new AutopilotPolicy(
                dto.enabled, allowed, dto.confidenceThreshold, dto.deterrenceTimeoutSeconds,
                new EscalationRules(dto.autoEscalateOnDeterrenceFailure,
                        dto.maxDeterrenceAttempts, alwaysEscalate),
                new SafetyConstraints(dto.requireMinimumConfidence,
                        dto.requireMultiModalConfirmation, neverAutonomous),
                safeAutopilotState(dto.syncState), parseInstant(dto.lastSyncTimestamp));
    }

    private static Set<ThreatType> parseThreatTypes(java.util.List<String> values) {
        EnumSet<ThreatType> result = EnumSet.noneOf(ThreatType.class);
        if (values != null) {
            for (String value : values) {
                ThreatType parsed = safeThreatType(value);
                if (parsed != null) result.add(parsed);
            }
        }
        return result;
    }

    private static ThreatType safeThreatType(String value) {
        if (value == null) return null;
        try { return ThreatType.valueOf(value); }
        catch (IllegalArgumentException ignored) { return null; }
    }

    private static Instant parseInstant(String value) {
        if (value == null || value.isBlank()) return null;
        try { return Instant.parse(value); }
        catch (RuntimeException ignored) {
            try { return java.time.OffsetDateTime.parse(value).toInstant(); }
            catch (RuntimeException ignoredAgain) { return null; }
        }
    }

    private static AutopilotState safeAutopilotState(String value) {
        if (value == null) return AutopilotState.DISABLED;
        try { return AutopilotState.valueOf(value); }
        catch (IllegalArgumentException ignored) { return AutopilotState.DISABLED; }
    }

    public AutopilotPolicyDto toDto(AutopilotPolicy p) {
        AutopilotPolicyDto dto = new AutopilotPolicyDto();
        dto.enabled = p.isEnabled();
        dto.allowedThreatTypes = p.getAllowedThreatTypes().stream().map(Enum::name).collect(Collectors.toList());
        dto.confidenceThreshold = p.getConfidenceThreshold();
        dto.deterrenceTimeoutSeconds = p.getDeterrenceTimeoutSeconds();
        dto.autoEscalateOnDeterrenceFailure = p.getEscalationRules().isAutoEscalateOnDeterrenceFailure();
        dto.maxDeterrenceAttempts = p.getEscalationRules().getMaxDeterrenceAttempts();
        dto.alwaysEscalateTypes = p.getEscalationRules().getAlwaysEscalateTypes().stream().map(Enum::name).collect(Collectors.toList());
        dto.requireMinimumConfidence = p.getSafetyConstraints().isRequireMinimumConfidence();
        dto.requireMultiModalConfirmation = p.getSafetyConstraints().isRequireMultiModalConfirmation();
        dto.neverAutonomousTypes = p.getSafetyConstraints().getNeverAutonomousTypes().stream().map(Enum::name).collect(Collectors.toList());
        dto.syncState = p.getSyncState() != null ? p.getSyncState().name() : null;
        dto.lastSyncTimestamp = p.getLastSyncTimestamp() != null ? p.getLastSyncTimestamp().toString() : null;
        return dto;
    }
}
