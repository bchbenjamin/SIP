package com.sip.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@EnableJpaRepositories("com.sip.backend.repository")
@EntityScan("com.sip.backend.domain") // Assuming a 'domain' package for entities later
public class SipBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(SipBackendApplication.class, args);
	}
}