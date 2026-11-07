package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;

public class LoginResponse {
    @SerializedName("token")
    public String token;

    @SerializedName("refreshToken")
    public String refreshToken;

    @SerializedName("expiresIn")
    public long expiresIn;

    @SerializedName("userId")
    public String userId;

    @SerializedName("username")
    public String username;

    @SerializedName("role")
    public String role;

    public LoginResponse() {}

    public LoginResponse(String token, String refreshToken, long expiresIn,
                         String userId, String username, String role) {
        this.token = token;
        this.refreshToken = refreshToken;
        this.expiresIn = expiresIn;
        this.userId = userId;
        this.username = username;
        this.role = role;
    }
}
