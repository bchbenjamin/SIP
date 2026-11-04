package com.sip.backend.auth;

import com.sip.backend.domain.User;
import com.sip.backend.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public AuthService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public Optional<User> findByUsername(String username) {
        return userRepository.findByUsername(username);
    }

    public String hashPassword(String rawPassword) {
        return passwordEncoder.encode(rawPassword);
    }

    // New method for refresh logic
    public boolean validateAndRotateToken(String refreshToken) {
        // TODO: Hash the incoming token, check against refresh_tokens table, and atomically revoke/issue new one
        return true; // Stub
    }
}

    // This method will be expanded to handle token generation based on the successful user lookup
}