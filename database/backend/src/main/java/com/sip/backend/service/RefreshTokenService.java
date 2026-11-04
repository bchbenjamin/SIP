package com.sip.backend.service;

import com.sip.backend.domain.RefreshToken;
import com.sip.backend.repository.RefreshTokenRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.ZonedDateTime;
import java.util.Optional;

@Service
public class RefreshTokenService {
    private final RefreshTokenRepository refreshTokenRepository;

    public RefreshTokenService(RefreshTokenRepository refreshTokenRepository) {
        this.refreshTokenRepository = refreshTokenRepository;
    }

    @Transactional
    public com.sip.backend.domain.RefreshToken createForUser(com.sip.backend.domain.User user) {
        // TODO: Generate a high-entropy, opaque refresh token string, hash it, and persist.
        // Stubbing for now.
        RefreshToken token = new RefreshToken();
        token.setUserId(user.getId());
        // token.setTokenHash(hash(rawToken));
        // token.setExpiresAt(ZonedDateTime.now().plus(7, ChronoUnit.DAYS));
        // token.setCreatedAt(ZonedDateTime.now());
        return token;
    }

    @Transactional
    public com.sip.backend.domain.RefreshToken consumeAndValidate(String rawToken) {
        // TODO: Hash rawToken, check DB, check expiry, return entity if valid.
        return null; // Stub
    }

    @Transactional
    public void revokeAllForUser(String userId) {
        // Logic: delete all tokens for this user ID
    }
}