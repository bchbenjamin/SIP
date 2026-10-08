package com.sip.guardian.data.repository;

import com.sip.guardian.data.local.SecureTokenStore;
import com.sip.guardian.data.remote.api.AuthApiService;
import com.sip.guardian.data.remote.dto.LoginRequest;
import com.sip.guardian.data.remote.dto.LoginResponse;
import com.sip.guardian.domain.model.User;
import com.sip.guardian.domain.repository.AuthRepository;

import java.io.IOException;
import java.util.Map;

import javax.inject.Inject;
import javax.inject.Singleton;

@Singleton
public class AuthRepositoryImpl implements AuthRepository {

    private final AuthApiService authApi;
    private final SecureTokenStore tokenStore;

    @Inject
    public AuthRepositoryImpl(AuthApiService authApi, SecureTokenStore tokenStore) {
        this.authApi = authApi;
        this.tokenStore = tokenStore;
    }

    @Override
    public User login(String username, String password) {
        try {
            var response = authApi.login(new LoginRequest(username, password)).execute();
            if (!response.isSuccessful() || response.body() == null) {
                if (response.code() == 401 || response.code() == 403) {
                    throw new SecurityException("Invalid username or password");
                }
                throw new IllegalStateException("Backend login failed (HTTP " + response.code() + ")");
            }
            persist(response.body());
            return currentUser();
        } catch (IOException e) {
            throw new IllegalStateException("Login failed: unable to reach the backend", e);
        }
    }

    @Override
    public User refreshSession() {
        String refresh = tokenStore.getRefreshToken();
        if (refresh == null) throw new SecurityException("No refresh token");
        try {
            var response = authApi.refresh(Map.of("refreshToken", refresh)).execute();
            if (!response.isSuccessful() || response.body() == null) {
                tokenStore.clear();
                throw new SecurityException("Session expired. Please sign in again.");
            }
            persist(response.body());
            return currentUser();
        } catch (IOException e) {
            throw new IllegalStateException("Session refresh failed: unable to reach the backend", e);
        }
    }

    @Override
    public void logout() {
        String accessToken = tokenStore.getAccessToken();
        String refreshToken = tokenStore.getRefreshToken();
        try {
            if (accessToken != null || refreshToken != null) {
                Map<String, String> body = refreshToken != null
                        ? Map.of("refreshToken", refreshToken) : Map.of();
                authApi.logout(accessToken != null ? "Bearer " + accessToken : null, body).execute();
            }
        } catch (IOException | RuntimeException ignored) {
            // Always clear local credentials even if the backend is unreachable.
        } finally {
            tokenStore.clear();
        }
    }

    @Override
    public User currentUser() {
        String id = tokenStore.getUserId();
        if (id == null) return null;
        String role = tokenStore.getRole();
        return new User(id, tokenStore.getUsername(),
                "ADMIN".equals(role) ? User.Role.ADMIN : User.Role.OPERATOR);
    }

    private void persist(LoginResponse tokens) {
        if (tokens == null || tokens.token == null || tokens.refreshToken == null
                || tokens.userId == null || tokens.username == null || tokens.role == null) {
            throw new IllegalStateException("Backend returned an incomplete login response");
        }
        tokenStore.save(tokens.token, tokens.refreshToken, tokens.expiresIn);
        tokenStore.saveUser(tokens.userId, tokens.username, tokens.role);
    }
}
