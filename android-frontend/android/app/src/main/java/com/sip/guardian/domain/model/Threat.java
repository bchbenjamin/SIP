package com.sip.guardian.domain.model;

/**
 * Classifies the nature and severity of a detected threat.
 * OOP: Abstraction — hides classification complexity behind type/severity.
 */
public final class Threat {
    private final ThreatType type;
    private final ThreatSeverity severity;
    private final String description;

    public Threat(ThreatType type, ThreatSeverity severity, String description) {
        this.type = type;
        this.severity = severity;
        this.description = description == null ? "" : description;
    }

    public ThreatType getType() { return type; }
    public ThreatSeverity getSeverity() { return severity; }
    public String getDescription() { return description; }

    public boolean isDeterrable() { return severity == ThreatSeverity.DETERRABLE; }

    @Override
    public String toString() {
        return "Threat{" + type + ", " + severity + ", '" + description + "'}";
    }
}
