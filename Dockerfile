# Test build - minimal
FROM alpine:latest
WORKDIR /app
RUN echo "hello world" > /app/test.txt
CMD cat /app/test.txt