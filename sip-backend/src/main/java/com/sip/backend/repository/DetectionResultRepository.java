package com.sip.backend.repository;

import com.sip.backend.domain.DetectionResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DetectionResultRepository extends JpaRepository<DetectionResult, String> {}