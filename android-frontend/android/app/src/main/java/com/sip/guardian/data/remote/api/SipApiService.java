package com.sip.guardian.data.remote.api;

import com.sip.guardian.data.remote.dto.AutopilotPolicyDto;
import com.sip.guardian.data.remote.dto.DashboardDto;
import com.sip.guardian.data.remote.dto.IncidentDto;
import com.sip.guardian.data.remote.dto.NodeDto;
import com.sip.guardian.data.remote.dto.PageDto;
import com.sip.guardian.data.remote.dto.VerificationRequest;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.PUT;
import retrofit2.http.Query;

/** Retrofit interface matching plan §13 API specification. */
public interface SipApiService {

    @GET("api/v1/incidents")
    Call<PageDto<IncidentDto>> getIncidents(
            @Query("page") Integer page,
            @Query("size") Integer size,
            @Query("state") String state,
            @Query("threatType") String threatType,
            @Query("nodeId") String nodeId,
            @Query("from") String from,
            @Query("to") String to);

    @GET("api/v1/incidents/{id}")
    Call<IncidentDto> getIncident(@Path("id") String id);

    @POST("api/v1/incidents/{id}/verify")
    Call<IncidentDto> verifyIncident(@Path("id") String id, @Body VerificationRequest request);

    @GET("api/v1/evidence/{incidentId}/image")
    Call<okhttp3.ResponseBody> getEvidenceImage(@Path("incidentId") String incidentId);

    @GET("api/v1/evidence/{incidentId}/video")
    Call<okhttp3.ResponseBody> getEvidenceVideo(@Path("incidentId") String incidentId);

    @GET("api/v1/evidence/{incidentId}/audio")
    Call<okhttp3.ResponseBody> getEvidenceAudio(@Path("incidentId") String incidentId);

    @GET("api/v1/evidence/{incidentId}/thumbnail")
    Call<okhttp3.ResponseBody> getEvidenceThumbnail(@Path("incidentId") String incidentId);

    @GET("api/v1/nodes")
    Call<List<NodeDto>> getNodes();

    @GET("api/v1/nodes/{id}")
    Call<NodeDto> getNode(@Path("id") String id);

    @GET("api/v1/autopilot/policy")
    Call<AutopilotPolicyDto> getAutopilotPolicy(@Query("nodeId") String nodeId);

    @PUT("api/v1/autopilot/policy")
    Call<AutopilotPolicyDto> updateAutopilotPolicy(
            @Query("nodeId") String nodeId,
            @Body AutopilotPolicyDto policy);

    @GET("api/v1/dashboard")
    Call<DashboardDto> getDashboard();

    @GET("api/v1/dataset/export")
    Call<okhttp3.ResponseBody> exportDataset(@Query("format") String format,
                                             @Query("from") String from,
                                             @Query("to") String to,
                                             @Query("labels") String labels);

    @GET("api/v1/dataset/stats")
    Call<java.util.Map<String, Object>> getDatasetStats();
}
