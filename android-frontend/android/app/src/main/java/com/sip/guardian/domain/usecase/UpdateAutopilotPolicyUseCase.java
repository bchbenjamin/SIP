package com.sip.guardian.domain.usecase;

import com.sip.guardian.domain.model.AutopilotPolicy;
import com.sip.guardian.domain.model.User;
import com.sip.guardian.domain.repository.AuthRepository;
import com.sip.guardian.domain.repository.AutopilotRepository;

import javax.inject.Inject;

public class UpdateAutopilotPolicyUseCase {
    private final AutopilotRepository repository;
    private final AuthRepository authRepository;

    @Inject
    public UpdateAutopilotPolicyUseCase(AutopilotRepository repository,
                                        AuthRepository authRepository) {
        this.repository = repository;
        this.authRepository = authRepository;
    }

    /** ADMIN-only for policy changes in prototype; relax when roles evolve (plan §26). */
    public AutopilotPolicy execute(String nodeId, AutopilotPolicy policy) {
        User user = authRepository.currentUser();
        if (user == null || !user.canManagePolicy()) {
            throw new SecurityException("Policy updates require ADMIN role");
        }
        return repository.updatePolicy(nodeId, policy);
    }
}
