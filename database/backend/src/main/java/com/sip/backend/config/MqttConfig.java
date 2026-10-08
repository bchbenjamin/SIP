package com.sip.backend.config;

import com.sip.backend.config.MqttConfigProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.integration.channel.DirectChannel;
import org.springframework.integration.core.MessageProducer;
import org.springframework.integration.mqtt.core.DefaultMqttPahoClientFactory;
import org.springframework.integration.mqtt.core.MqttPahoClientFactory;
import org.springframework.integration.mqtt.inbound.MqttPahoMessageReceiver;
import org.springframework.integration.mqtt.outbound.MqttPahoMessageHandler;
import org.springframework.integration.mqtt.support.DefaultMqttHeaderMapper;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.core.MessageHandler;

@Configuration
@ConditionalOnProperty(name = "app.mqtt.enabled", havingValue = "true", matchIfMissing = false)
public class MqttConfig {

    public static final String MQTT_INBOUND_CHANNEL = "mqttInboundChannel";
    public static final String MQTT_OUTBOUND_CHANNEL = "mqttOutboundChannel";
    public static final String MQTT_COMMAND_CHANNEL = "mqttCommandChannel";

    @Bean
    public MqttPahoClientFactory mqttClientFactory(MqttConfigProperties props) {
        DefaultMqttPahoClientFactory factory = new DefaultMqttPahoClientFactory();
        factory.setServerURIs(props.getBrokerUrl());
        if (props.getUsername() != null && !props.getUsername().isBlank()) {
            factory.setUserName(props.getUsername());
        }
        if (props.getPassword() != null && !props.getPassword().isBlank()) {
            factory.setPassword(props.getPassword());
        }
        factory.setKeepAliveInterval(props.getKeepAliveInterval());
        factory.setConnectionTimeout(props.getCommandTimeout());
        return factory;
    }

    @Bean
    public MessageChannel mqttInboundChannel() {
        return new DirectChannel();
    }

    @Bean
    public MessageChannel mqttOutboundChannel() {
        return new DirectChannel();
    }

    @Bean
    public MessageChannel mqttCommandChannel() {
        return new DirectChannel();
    }

    @Bean
    public MessageProducer mqttInbound(MqttPahoClientFactory factory, MqttConfigProperties props) {
        MqttPahoMessageReceiver receiver = new MqttPahoMessageReceiver();
        receiver.setClientFactory(factory);
        receiver.setClientId(props.getClientId() + "-inbound");
        receiver.setOutputChannel(mqttInboundChannel());
        receiver.setTopicExpression(payload -> "sip/+/incidents");
        receiver.setHeaderMapper(new DefaultMqttHeaderMapper());
        return receiver;
    }

    @Bean
    public MessageHandler mqttOutbound(MqttPahoClientFactory factory, MqttConfigProperties props) {
        MqttPahoMessageHandler handler = new MqttPahoMessageHandler();
        handler.setClientFactory(factory);
        handler.setDefaultTopic("sip");
        handler.setAsync(true);
        return handler;
    }
}