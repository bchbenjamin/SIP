FROM eclipse-temurin:17-jre-alpine
WORKDIR /app

RUN apk add --no-cache curl

COPY database/backend/sip-backend-0.1.0.jar app.jar

EXPOSE 10000

# Health check before starting
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:10000/actuator/health || exit 1

ENTRYPOINT ["sh", "-c", "java -XX:+UseG1GC -Xmx512m -Djava.security.egd=file:/dev/./urandom -jar app.jar"]