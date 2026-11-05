package com.sip.guardian.data.repository;

import com.sip.guardian.data.local.SecureTokenStore;
import com.sip.guardian.data.remote.api.AuthApiService;
import com.sip.guardian.domain.model.User;

import okhttp3.OkHttpClient;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import java.io.IOException;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

public class AuthRepositoryTest {

    private MockWebServer server;
    private AuthApiService authApi;
    private SecureTokenStore tokenStore;
    private AuthRepositoryImpl repository;

    @Before
    public void setUp() throws IOException {
        server = new MockWebServer();
        server.start();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(server.url("/"))
                .client(new OkHttpClient())
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        authApi = retrofit.create(AuthApiService.class);
        tokenStore = Mockito.mock(SecureTokenStore.class);
        repository = new AuthRepositoryImpl(authApi, tokenStore);
    }

    @After
    public void tearDown() throws IOException {
        server.shutdown();
    }

    @Test
    public void loginSuccessfulResponseParsedAndPersisted() {
        String jsonResponse = "{\n" +
                "  \"token\": \"mock-jwt-token\",\n" +
                "  \"refreshToken\": \"mock-refresh-token\",\n" +
                "  \"expiresIn\": 900,\n" +
                "  \"userId\": \"user-123\",\n" +
                "  \"username\": \"testuser\",\n" +
                "  \"role\": \"ADMIN\"\n" +
                "}";

        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        when(tokenStore.getUserId()).thenReturn("user-123");
        when(tokenStore.getUsername()).thenReturn("testuser");
        when(tokenStore.getRole()).thenReturn("ADMIN");

        User user = repository.login("testuser", "password123");

        assertNotNull(user);
        assertEquals("user-123", user.getId());
        assertEquals("testuser", user.getUsername());
        assertEquals(User.Role.ADMIN, user.getRole());

        verify(tokenStore).save("mock-jwt-token", "mock-refresh-token", 900);
        verify(tokenStore).saveUser("user-123", "testuser", "ADMIN");
    }

    @Test(expected = SecurityException.class)
    public void loginUnauthorizedThrowsSecurityException() {
        server.enqueue(new MockResponse()
                .setResponseCode(401)
                .setBody("{\"error\": \"Unauthorized\"}"));

        repository.login("wronguser", "badpass");
    }

    @Test(expected = SecurityException.class)
    public void loginForbiddenThrowsSecurityException() {
        server.enqueue(new MockResponse()
                .setResponseCode(403)
                .setBody("{\"error\": \"Forbidden\"}"));

        repository.login("forbiddenUser", "badpass");
    }

    @Test(expected = IllegalStateException.class)
    public void loginServerErrorThrowsIllegalStateException() {
        server.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("{\"error\": \"Internal Server Error\"}"));

        repository.login("anyuser", "anypass");
    }

    @Test(expected = IllegalStateException.class)
    public void loginIncompleteResponseThrowsIllegalStateException() {
        // Missing token and refreshToken
        String jsonResponse = "{\n" +
                "  \"userId\": \"user-123\",\n" +
                "  \"username\": \"testuser\"\n" +
                "}";

        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        repository.login("testuser", "password123");
    }

    @Test(expected = IllegalStateException.class)
    public void loginMalformedJsonThrowsIllegalStateException() {
        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setBody("{ not valid json }")
                .addHeader("Content-Type", "application/json"));

        repository.login("testuser", "password123");
    }

    @Test
    public void refreshSessionSuccessful() {
        when(tokenStore.getRefreshToken()).thenReturn("existing-refresh-token");
        when(tokenStore.getUserId()).thenReturn("user-123");
        when(tokenStore.getUsername()).thenReturn("testuser");
        when(tokenStore.getRole()).thenReturn("OPERATOR");

        String jsonResponse = "{\n" +
                "  \"token\": \"new-jwt-token\",\n" +
                "  \"refreshToken\": \"new-refresh-token\",\n" +
                "  \"expiresIn\": 900,\n" +
                "  \"userId\": \"user-123\",\n" +
                "  \"username\": \"testuser\",\n" +
                "  \"role\": \"OPERATOR\"\n" +
                "}";

        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        User user = repository.refreshSession();

        assertNotNull(user);
        assertEquals("user-123", user.getId());
        assertEquals("testuser", user.getUsername());
        assertEquals(User.Role.OPERATOR, user.getRole());

        verify(tokenStore).save("new-jwt-token", "new-refresh-token", 900);
        verify(tokenStore).saveUser("user-123", "testuser", "OPERATOR");
    }

    @Test(expected = SecurityException.class)
    public void refreshSessionNoTokenThrowsSecurityException() {
        when(tokenStore.getRefreshToken()).thenReturn(null);
        repository.refreshSession();
    }

    @Test(expected = SecurityException.class)
    public void refreshSessionExpiredThrowsSecurityExceptionAndClearsStore() {
        when(tokenStore.getRefreshToken()).thenReturn("expired-refresh-token");
        server.enqueue(new MockResponse().setResponseCode(401));

        try {
            repository.refreshSession();
        } finally {
            verify(tokenStore).clear();
        }
    }

    @Test
    public void logoutClearsCredentialsEvenWhenServerFails() {
        when(tokenStore.getAccessToken()).thenReturn("token-to-logout");
        when(tokenStore.getRefreshToken()).thenReturn("refresh-to-logout");
        server.enqueue(new MockResponse().setResponseCode(500));

        repository.logout();

        verify(tokenStore).clear();
    }

    @Test
    public void subsequentLoginAfterLogoutSucceeds() {
        // 1. Logout
        when(tokenStore.getAccessToken()).thenReturn("old-token");
        server.enqueue(new MockResponse().setResponseCode(200));
        repository.logout();
        verify(tokenStore).clear();

        // 2. Subsequent Login
        String jsonResponse = "{\n" +
                "  \"token\": \"fresh-jwt-token\",\n" +
                "  \"refreshToken\": \"fresh-refresh-token\",\n" +
                "  \"expiresIn\": 900,\n" +
                "  \"userId\": \"user-456\",\n" +
                "  \"username\": \"newuser\",\n" +
                "  \"role\": \"ADMIN\"\n" +
                "}";

        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        when(tokenStore.getUserId()).thenReturn("user-456");
        when(tokenStore.getUsername()).thenReturn("newuser");
        when(tokenStore.getRole()).thenReturn("ADMIN");

        User user = repository.login("newuser", "securepass");
        assertNotNull(user);
        assertEquals("user-456", user.getId());
        assertEquals("newuser", user.getUsername());
        verify(tokenStore).save("fresh-jwt-token", "fresh-refresh-token", 900);
        verify(tokenStore).saveUser("user-456", "newuser", "ADMIN");
    }
}
