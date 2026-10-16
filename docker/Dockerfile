FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY database/backend/sip-backend-0.1.0.jar app.jar
EXPOSE 10000
ENTRYPOINT ["sh", "-c", "java -XX:+UseG1GC -Xmx512m -jar app.jar"]