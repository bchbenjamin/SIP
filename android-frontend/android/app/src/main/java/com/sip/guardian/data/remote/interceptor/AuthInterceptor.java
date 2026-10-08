package com.sip.guardian.data.remote.interceptor;

import com.sip.guardian.data.local.SecureTokenStore;

import java.io.IOException;

import javax.inject.Inject;
import javax.inject.Singleton;

import okhttp3.Interceptor;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Injects the JWT bearer token into every API request.
 * OOP: Decorator pattern (OkHttp Interceptor). Never logs the token.
 */
@Singleton
public class AuthInterceptor implements Interceptor {

    private final SecureTokenStore tokenStore;

    @Inject
    public AuthInterceptor(SecureTokenStore tokenStore) {
        this.tokenStore = tokenStore;
    }

    @Override
    public Response intercept(Chain chain) throws IOException {
        Request original = chain.request();
        String token = tokenStore.getAccessToken();

        if (token == null || original.header("Authorization") != null) {
            return chain.proceed(original);
        }

        Request authorized = original.newBuilder()
                .header("Authorization", "Bearer " + token)
                .build();
        return chain.proceed(authorized);
    }
}
