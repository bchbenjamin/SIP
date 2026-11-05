FROM docker.io/library/eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY database/backend/sip-backend-0.1.0.jar app.jar
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["sh", "-c", "java -XX:+UseG1GC -Xmx512m -Djava.security.egd=file:/dev/./urandom -jar app.jar"]
