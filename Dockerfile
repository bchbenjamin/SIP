# Minimal Docker build for SIP Backend
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY database/backend/sip-backend-0.1.0.jar app.jar
EXPOSE 8080
ENV JAVA_OPTS="-XX:+UseG1GC -Xmx512m -Djava.security.egd=file:/dev/./urandom"
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]