package com.sip.guardian.data.repository;

import com.sip.guardian.data.mapper.AutopilotMapper;
import com.sip.guardian.data.remote.api.SipApiService;
import com.sip.guardian.data.remote.dto.AutopilotPolicyDto;
import com.sip.guardian.domain.model.AutopilotPolicy;
import com.sip.guardian.domain.repository.AutopilotRepository;

import java.io.IOException;

import javax.inject.Inject;
import javax.inject.Singleton;

import retrofit2.Response;

@Singleton
public class AutopilotRepositoryImpl implements AutopilotRepository {

    private final SipApiService api;
    private final AutopilotMapper mapper;

    @Inject
    public AutopilotRepositoryImpl(SipApiService api, AutopilotMapper mapper) {
        this.api = api;
        this.mapper = mapper;
    }

    @Override
    public AutopilotPolicy getPolicy(String nodeId) {
        try {
            Response<AutopilotPolicyDto> r = api.getAutopilotPolicy(nodeId).execute();
            if (r.isSuccessful() && r.body() != null) return mapper.toDomain(r.body());
        } catch (IOException ignored) { }
        return AutopilotPolicy.disabled();
    }

    @Override
    public AutopilotPolicy updatePolicy(String nodeId, AutopilotPolicy policy) {
        try {
            Response<AutopilotPolicyDto> r =
                    api.updateAutopilotPolicy(nodeId, mapper.toDto(policy)).execute();
            if (r.isSuccessful() && r.body() != null) return mapper.toDomain(r.body());
            throw new IllegalStateException("Policy update failed: HTTP " + r.code());
        } catch (IOException e) {
            throw new IllegalStateException("Policy update failed: network error", e);
        }
    }
}
