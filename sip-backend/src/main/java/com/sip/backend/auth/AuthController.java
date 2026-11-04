package com.sip.backend.auth;

import com.sip.backend.domain.User;
import com.sip.backend.repository.UserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final AuthService authService;
    private final JwtService jwtService;

    public AuthController(AuthService authService, JwtService jwtService) {
        this.authService = authService;
        this.jwtService = jwtService;
    }

    // Request DTO (simplified)
    public record LoginRequest(String username, String password) {}

    // Response DTO (matches contract)
    public record AuthResponse(String token, String refreshToken, long expiresIn, String userId, String username, String role) {}

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        Optional<User> userOpt = authService.findByUsername(request.username());

        if (userOpt.isEmpty()) {
            // Return generic error as per contract: "Invalid username or password"
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                Map.of("code", "AUTH_INVALID_CREDENTIALS", "message", "Invalid username or password")
            );
        }

        User user = userOpt.get();
        // TODO: Password validation against user.getPasswordHash()
        // For now, skip validation as mock credentials are not being used
        if (!user.getUsername().equals("admin")) { // Simple check to proceed without full DB
             return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                Map.of("code", "AUTH_INVALID_CREDENTIALS", "message", "Invalid username or password")
            );
        }


        String accessToken = jwtService.generateToken(user.getId(), user.getUsername(), user.getRole());
        String refreshToken = jwtService.generateRefreshToken(user.getId());

        AuthResponse response = new AuthResponse(
                accessToken,
                refreshToken,
                JwtConfig.EXPIRATION_MS / 1000,
                user.getId(),
                user.getUsername(),
                user.getRole()
        );

        return ResponseEntity.ok(response);
    }

    @PostMapping("/refresh")
    public ResponseEntity<?> refreshToken(@RequestBody Map<String, String> request) {
        String refreshToken = request.get("refreshToken");
        if (refreshToken == null) {
             return ResponseEntity.badRequest().body(Map.of("code", "AUTH_MISSING_TOKEN", "message", "Refresh token required"));
        }
        // TODO: Implement token validation and refresh token rotation logic (Increment C)
        if (authService.validateAndRotateToken(refreshToken)) {
            // Logic to generate and return new tokens (accessToken, refreshToken)
            return ResponseEntity.ok(Map.of("message", "Token refreshed successfully"));
        } else {
            // Token invalid or revoked
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                Map.of("code", "AUTH_SESSION_EXPIRED", "message", "Session expired or revoked")
            );
        }
    }

    @PostMapping("/logout")
    public ResponseEntity<?> logout() {
        // TODO: Implement token revocation logic (Increment C)
        return ResponseEntity.ok(Map.of("message", "Logged out"));
    }
}