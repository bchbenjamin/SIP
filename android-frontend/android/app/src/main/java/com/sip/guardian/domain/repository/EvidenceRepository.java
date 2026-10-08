package com.sip.guardian.domain.repository;

import com.sip.guardian.domain.model.EvidenceBundle;

import java.io.File;

/** Fetches evidence binaries from authenticated gateway endpoints (plan §16). */
public interface EvidenceRepository {
    /** Local cached file for an evidence URL; downloads on miss. */
    File getEvidenceFile(String incidentId, EvidenceBundle bundle, com.sip.guardian.domain.model.EvidenceType type);
}
