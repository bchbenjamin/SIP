package com.sip.guardian.domain.usecase;

import com.sip.guardian.domain.model.HumanAnnotation;
import com.sip.guardian.domain.model.HumanLabel;
import com.sip.guardian.domain.model.Incident;
import com.sip.guardian.domain.model.User;
import com.sip.guardian.domain.repository.AuthRepository;
import com.sip.guardian.domain.repository.IncidentRepository;

import java.time.Instant;
import java.util.UUID;

import javax.inject.Inject;

/**
 * Business logic for the human verification workflow (plan §6).
 * OOP: Single Responsibility + Dependency Inversion.
 */
public class VerifyIncidentUseCase {

    private final IncidentRepository incidentRepository;
    private final AuthRepository authRepository;

    @Inject
    public VerifyIncidentUseCase(IncidentRepository incidentRepository,
                                 AuthRepository authRepository) {
        this.incidentRepository = incidentRepository;
        this.authRepository = authRepository;
    }

    public Result execute(String incidentId, HumanLabel label, String notes) {
        User user = authRepository.currentUser();
        if (user == null) return Result.failure("Not authenticated");
        if (!user.canVerify()) return Result.failure("User not authorized to verify");

        Incident incident = incidentRepository.getIncidentById(incidentId);
        if (incident == null) return Result.failure("Incident not found");
        if (!incident.requiresVerification()) {
            return Result.failure("Incident is not pending verification (state="
                    + incident.getState() + ")");
        }

        HumanAnnotation annotation = new HumanAnnotation(
                UUID.randomUUID().toString(),
                label,
                user.getId(),
                Instant.now(),
                notes,
                null,   // operator may leave confidence unset
                1);

        Incident updated = incidentRepository.verifyIncident(incidentId, annotation);
        return Result.success(updated);
    }

    /** Simple Result wrapper so the presentation layer never sees exceptions. */
    public static final class Result {
        private final Incident incident;
        private final String error;

        private Result(Incident incident, String error) {
            this.incident = incident;
            this.error = error;
        }

        static Result success(Incident i) { return new Result(i, null); }
        static Result failure(String e) { return new Result(null, e); }

        public boolean isSuccess() { return error == null; }
        public Incident getIncident() { return incident; }
        public String getError() { return error; }
    }
}
