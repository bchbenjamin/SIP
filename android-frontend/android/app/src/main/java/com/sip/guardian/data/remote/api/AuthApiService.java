package com.sip.guardian.data.remote.api;

import com.sip.guardian.data.remote.dto.LoginRequest;
import com.sip.guardian.data.remote.dto.LoginResponse;

import java.util.Map;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Header;
import retrofit2.http.POST;

public interface AuthApiService {
    @POST("api/v1/auth/login")
    Call<LoginResponse> login(@Body LoginRequest request);

    @POST("api/v1/auth/refresh")
    Call<LoginResponse> refresh(@Body Map<String, String> body);

    @POST("api/v1/auth/logout")
    Call<Void> logout(@Header("Authorization") String authorization, @Body Map<String, String> body);
}
