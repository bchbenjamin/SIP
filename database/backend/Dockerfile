# ─── Build stage ─────────────────────────────────────────────────────────────
FROM eclipse-temurin:17-jdk-alpine AS builder

WORKDIR /app

# Copy Gradle wrapper and build config first (for dependency caching)
COPY gradlew gradle/wrapper/gradle-wrapper.properties ./
COPY build.gradle settings.gradle ./

# Cache Gradle dependencies
RUN chmod +x ./gradlew && ./gradlew dependencies --no-daemon

# Copy source and build
COPY src src
COPY gradle gradle

RUN ./gradlew bootJar --no-daemon

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Add non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY --from=builder /app/build/libs/sip-backend-0.1.0.jar app.jar

RUN chown appuser:appgroup app.jar

USER appuser

EXPOSE 8080

# Production JVM tuning: use container memory limits, G1GC, disable unnecessary checks
ENV JAVA_OPTS="-XX:+UseG1GC \
    -XX:MaxRAMPercentage=75.0 \
    -XX:InitialRAMPercentage=50.0 \
    -XX:+UseContainerSupport \
    -XX:+ExitOnOutOfMemoryError \
    -Djava.security.egd=file:/dev/./urandom"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]