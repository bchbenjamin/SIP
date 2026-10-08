package com.sip.guardian.domain.repository;

import java.io.File;
import java.time.Instant;
import java.util.Map;

/** Dataset export for model retraining (YOLO first, exporter extensible to COCO). */
public interface DatasetRepository {
    /** Downloads a ZIP of images+labels in the requested format. */
    File export(String format, Instant from, Instant to, String labels);

    Map<String, Object> stats();
}
