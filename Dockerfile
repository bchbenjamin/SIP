FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY database/backend/sip-backend-0.1.0.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-XX:+UseG1GC", "-Xmx512m", "-Djava.security.egd=file:/dev/./urandom", "-jar", "app.jar"]