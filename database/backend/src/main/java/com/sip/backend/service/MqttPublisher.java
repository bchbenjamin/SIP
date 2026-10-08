package com.sip.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sip.backend.config.MqttConfig;
import com.sip.backend.entity.Command;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.integration.mqtt.core.MqttPahoClientFactory;
import org.springframework.integration.mqtt.outbound.MqttPahoMessageHandler;
import org.springframework.messaging.Message;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@ConditionalOnProperty(name = "app.mqtt.enabled", havingValue = "true", matchIfMissing = false)
public class MqttPublisher {

    private static final Logger log = LoggerFactory.getLogger(MqttPublisher.class);
    private static final int QOS = 1;

    private final MqttPahoMessageHandler handler;
    private final ObjectMapper objectMapper;

    public MqttPublisher(MqttPahoClientFactory factory, ObjectMapper objectMapper) {
        this.handler = new MqttPahoMessageHandler();
        this.handler.setClientFactory(factory);
        this.handler.setDefaultTopic("sip");
        this.handler.setAsync(true);
        this.objectMapper = objectMapper;
    }

    public void publishCommand(String nodeId, Command command) {
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("commandId", command.getId());
            payload.put("type", command.getCommandType().name());
            payload.put("priority", command.getPriority());
            payload.put("createdAt", command.getCreatedAt().toString());
            payload.put("payload", command.getPayload());
            if (command.getExpiresAt() != null) {
                payload.put("expiresAt", command.getExpiresAt().toString());
            }

            String json = objectMapper.writeValueAsString(payload);
            String topic = "sip/" + nodeId + "/commands";

            Message<String> message = MessageBuilder
                    .withPayload(json)
                    .setHeader("mqtt_topic", topic)
                    .setHeader("mqtt_qos", QOS)
                    .build();

            handler.handleMessage(message);
            log.debug("Published command {} to topic {}", command.getId(), topic);
        } catch (Exception e) {
            log.error("Failed to publish command {} to node {}: {}", command.getId(), nodeId, e.getMessage());
        }
    }

    public void publishIncidentAcknowledgement(String nodeId, String incidentId, boolean success) {
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("incidentId", incidentId);
            payload.put("success", success);

            String json = objectMapper.writeValueAsString(payload);
            String topic = "sip/" + nodeId + "/ack";

            Message<String> message = MessageBuilder
                    .withPayload(json)
                    .setHeader("mqtt_topic", topic)
                    .setHeader("mqtt_qos", 0)
                    .build();

            handler.handleMessage(message);
        } catch (Exception e) {
            log.error("Failed to publish ack for incident {} to node {}: {}", incidentId, nodeId, e.getMessage());
        }
    }
}