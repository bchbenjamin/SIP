package com.sip.backend;

import com.sip.backend.config.DataSourceUriParser;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.ConfigurableEnvironment;

/**
 * Registers DataSourceUriParser as a context initializer so DATABASE_URL is
 * parsed and its components are injected into the Spring Environment before
 * any bean — including the datasource — is created.
 */
@SpringBootApplication
public class SipBackendApplication {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(SipBackendApplication.class);

        // Add initializer that runs BEFORE context refresh
        app.addInitializers((ConfigurableApplicationContext ctx) -> {
            ConfigurableEnvironment env = ctx.getEnvironment();
            DataSourceUriParser.parseAndSet(env);
        });

        app.run(args);
    }
}