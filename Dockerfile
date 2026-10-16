# SIP Backend Docker container
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Copy pre-built JAR
COPY database/backend/sip-backend-0.1.0.jar app.jar

EXPOSE 10000

ENV JAVA_OPTS="-XX:+UseG1GC -Xmx512m -Djava.security.egd=file:/dev/./urandom"
ENV SERVER_PORT="10000"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]