package com.sip.guardian.data.remote.interceptor;

import com.sip.guardian.data.local.SecureTokenStore;
import com.sip.guardian.data.remote.api.AuthApiService;
import com.sip.guardian.data.remote.dto.LoginResponse;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.inject.Inject;
import javax.inject.Singleton;

import okhttp3.Authenticator;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.Route;

/**
 * Handles 401s by refreshing the access token once and retrying (plan §26).
 * Synchronized to prevent parallel refresh stampedes.
 */
@Singleton
public class TokenRefreshAuthenticator implements Authenticator {

    private final AuthApiService authApi;
    private final SecureTokenStore tokenStore;

    @Inject
    public TokenRefreshAuthenticator(AuthApiService authApi, SecureTokenStore tokenStore) {
        this.authApi = authApi;
        this.tokenStore = tokenStore;
    }

    @Override
    public synchronized Request authenticate(Route route, Response response) throws IOException {
        if (responseCount(response) >= 2) return null; // already retried with new token

        String refreshToken = tokenStore.getRefreshToken();
        if (refreshToken == null) return null;

        Map<String, String> body = new HashMap<>();
        body.put("refreshToken", refreshToken);
        retrofit2.Response<LoginResponse> refreshResponse =
                authApi.refresh(body).execute();

        if (!refreshResponse.isSuccessful() || refreshResponse.body() == null) {
            tokenStore.clear();
            return null;
        }

        LoginResponse tokens = refreshResponse.body();
        tokenStore.save(tokens.token, tokens.refreshToken, tokens.expiresIn);

        return response.request().newBuilder()
                .header("Authorization", "Bearer " + tokens.token)
                .build();
    }

    private int responseCount(Response response) {
        int count = 1;
        while ((response = response.priorResponse()) != null) count++;
        return count;
    }
}
