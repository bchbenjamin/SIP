# Multi-stage Docker build for SIP Backend
# Stage 1: Build the JAR
FROM eclipse-temurin:17-jdk-alpine AS builder

RUN apk add --no-cache curl unzip bash

WORKDIR /app

# Download and install Gradle 8.5 (instead of relying on gradle-wrapper.jar)
RUN curl -fsSL https://services.gradle.org/distributions/gradle-8.5-bin.zip -o gradle.zip \
    && unzip -q gradle.zip \
    && rm gradle.zip \
    && mv gradle-8.5 /opt/gradle
ENV PATH="/opt/gradle/bin:$PATH"

# Copy build configuration (from build context = database/backend/)
COPY build.gradle settings.gradle ./

# Download dependencies (layer cacheable)
RUN gradle dependencies --no-daemon --stacktrace

# Copy source and build JAR
COPY src src
RUN gradle clean bootJar --no-daemon --stacktrace

# Stage 2: Runtime
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY --from=builder /app/build/libs/sip-backend-0.1.0.jar app.jar

RUN chown appuser:appgroup app.jar

USER appuser

EXPOSE 8080

ENV JAVA_OPTS="-XX:+UseG1GC -XX:MaxRAMPercentage=75.0 -XX:InitialRAMPercentage=50.0 -XX:+UseContainerSupport -XX:+ExitOnOutOfMemoryError -Djava.security.egd=file:/dev/./urandom"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]