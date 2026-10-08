package com.sip.guardian.data.local;

import android.content.Context;
import android.content.SharedPreferences;
import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;
import java.io.IOException;
import java.security.GeneralSecurityException;
import javax.inject.Inject;
import javax.inject.Singleton;
import dagger.hilt.android.qualifiers.ApplicationContext;

@Singleton
public class SecureTokenStore {
    private static final String FILE = "sip_tokens";
    private static final String KEY_ACCESS = "access_token";
    private static final String KEY_REFRESH = "refresh_token";
    private static final String KEY_EXPIRES = "expires_at_epoch_ms";
    private static final String KEY_USER_ID = "user_id";
    private static final String KEY_USERNAME = "username";
    private static final String KEY_ROLE = "role";

    private final SharedPreferences prefs;

    @Inject
    public SecureTokenStore(@ApplicationContext Context context) {
        this.prefs = create(context);
    }

    private static SharedPreferences create(Context context) {
        try {
            MasterKey masterKey = new MasterKey.Builder(context)
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                    .build();
            return EncryptedSharedPreferences.create(
                    context, FILE, masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM);
        } catch (GeneralSecurityException | IOException e) {
            throw new IllegalStateException("Failed to initialize encrypted token store", e);
        }
    }

    public synchronized void save(String accessToken, String refreshToken, long expiresInSeconds) {
        if (accessToken == null || refreshToken == null) {
            throw new IllegalArgumentException("Tokens must not be null");
        }
        prefs.edit()
                .putString(KEY_ACCESS, accessToken)
                .putString(KEY_REFRESH, refreshToken)
                .putLong(KEY_EXPIRES,
                        System.currentTimeMillis() + Math.max(0, expiresInSeconds) * 1000)
                .apply();
    }

    public synchronized void saveUser(String userId, String username, String role) {
        prefs.edit()
                .putString(KEY_USER_ID, userId)
                .putString(KEY_USERNAME, username)
                .putString(KEY_ROLE, role)
                .apply();
    }

    public String getAccessToken() {
        if (isExpired()) return null;
        return prefs.getString(KEY_ACCESS, null);
    }

    public String getRefreshToken() { return prefs.getString(KEY_REFRESH, null); }

    public boolean isExpired() {
        long expiresAt = prefs.getLong(KEY_EXPIRES, 0);
        return expiresAt == 0 || System.currentTimeMillis() >= expiresAt - 30_000;
    }

    public String getUserId() { return prefs.getString(KEY_USER_ID, null); }
    public String getUsername() { return prefs.getString(KEY_USERNAME, null); }
    public String getRole() { return prefs.getString(KEY_ROLE, null); }

    public synchronized void clear() {
        prefs.edit().clear().apply();
    }
}
