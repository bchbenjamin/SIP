# ─── Build stage ─────────────────────────────────────────────────────────────
FROM eclipse-temurin:17-jdk-alpine AS builder

WORKDIR /app

# Install Gradle (no wrapper JAR needed — use distribution directly)
RUN apk add --no-cache curl unzip \
    && curl -fsL https://services.gradle.org/distributions/gradle-8.5-bin.zip -o gradle.zip \
    && unzip -q gradle.zip \
    && rm gradle.zip
ENV PATH="/app/gradle-8.5/bin:$PATH"

# Copy build config
COPY build.gradle settings.gradle ./

# Cache Gradle dependencies (layer cacheable)
RUN gradle dependencies --no-daemon

# Copy source and build
COPY src src
COPY gradle gradle

RUN gradle bootJar --no-daemon

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY --from=builder /app/build/libs/sip-backend-0.1.0.jar app.jar

RUN chown appuser:appgroup app.jar

USER appuser

EXPOSE 8080

ENV JAVA_OPTS="-XX:+UseG1GC -XX:MaxRAMPercentage=75.0 -XX:InitialRAMPercentage=50.0 -XX:+UseContainerSupport -XX:+ExitOnOutOfMemoryError -Djava.security.egd=file:/dev/./urandom"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]