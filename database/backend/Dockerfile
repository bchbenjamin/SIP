# Multi-stage Docker build for SIP Backend — optimized for Render free tier
# Build stage: minimal footprint, caches dependency downloads in Docker layers
FROM eclipse-temurin:17-jre-alpine AS builder

RUN apk add --no-cache curl unzip bash

WORKDIR /app

# Download Gradle once (layer cached on re-builds)
RUN curl -fsSL https://services.gradle.org/distributions/gradle-8.5-bin.zip -o gradle.zip \
    && unzip -q gradle.zip && rm gradle.zip && mv gradle-8.5 /opt/gradle
ENV PATH="/opt/gradle/bin:$PATH"

# Copy build config only — Gradle caches dependencies in /app/.gradle
COPY build.gradle settings.gradle ./

# Pre-fetch ALL dependencies into Docker layer (cached until build.gradle changes)
RUN gradle dependencies --no-daemon

# Copy source
COPY src src

# Build JAR (incremental — only recompiles changed classes)
RUN gradle bootJar --no-daemon

# Runtime stage: minimal, only the JRE + JAR
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY --from=builder /app/build/libs/sip-backend-0.1.0.jar app.jar

RUN chown appuser:appgroup app.jar

USER appuser

EXPOSE 8080

ENV JAVA_OPTS="-XX:+UseG1GC -XX:MaxRAMPercentage=75.0 -XX:InitialRAMPercentage=50.0 -XX:+UseContainerSupport -XX:+ExitOnOutOfMemoryError -Djava.security.egd=file:/dev/./urandom"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]