package com.sip.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
// JWT filter implementation will be added later

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    // JWT Filter and AuthenticationProvider bean dependencies will be injected here later

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                // Public endpoints: Login, Refresh
                .requestMatchers("/api/v1/auth/login", "/api/v1/auth/refresh").permitAll()
                // Actuator/Health endpoint is public for monitoring
                .requestMatchers("/actuator/health").permitAll()
                // Other endpoints require authentication
                .anyRequest().authenticated()
            );

        // Placeholder for adding JWT filter
        // http.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
        .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
    }
}