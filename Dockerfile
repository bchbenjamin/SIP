FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
RUN apk add --no-cache curl
COPY database/backend/sip-backend-0.1.0.jar app.jar
EXPOSE 10000
ENV SERVER_PORT=10000
# No HEALTHCHECK — let Render manage it
ENTRYPOINT ["sh", "-c", "java -XX:+UseG1GC -Xmx512m -Djava.security.egd=file:/dev/./urandom -jar app.jar"]