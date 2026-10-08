package com.sip.guardian.service;

import android.content.Context;

import com.sip.guardian.domain.model.EvidenceBundle;
import com.sip.guardian.domain.model.EvidenceType;
import com.sip.guardian.domain.repository.EvidenceRepository;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.inject.Inject;
import javax.inject.Singleton;

/**
 * Background pre-fetch of evidence (thumbnail-first, plan §16/§17)
 * so IncidentDetailScreen opens instantly.
 */
@Singleton
public class EvidenceCacheService {

    private final EvidenceRepository evidenceRepository;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);

    @Inject
    public EvidenceCacheService(EvidenceRepository evidenceRepository) {
        this.evidenceRepository = evidenceRepository;
    }

    public void prefetch(String incidentId, EvidenceBundle bundle) {
        if (bundle == null) return;
        for (EvidenceType type : bundle.getAvailableTypes()) {
            executor.submit(() -> {
                try {
                    File f = evidenceRepository.getEvidenceFile(incidentId, bundle, type);
                    // file now in cache dir; Coil/ExoPlayer read from it
                } catch (Exception ignored) {
                    // prefetch is best-effort only
                }
            });
        }
    }
}
