package com.sip.guardian.data.remote.websocket;

import android.os.Handler;
import android.os.Looper;

import com.sip.guardian.data.local.SecureTokenStore;
import com.sip.guardian.data.remote.api.AuthApiService;
import com.sip.guardian.data.remote.dto.LoginResponse;

import java.io.IOException;
import java.util.Map;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.inject.Inject;
import javax.inject.Singleton;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

/** Authenticated WebSocket client with bounded exponential-backoff reconnect. */
@Singleton
public class SipWebSocketClient {

    public interface Listener { void onEvent(WebSocketEvent event); }

    private static final long MIN_BACKOFF_MS = 1_000;
    private static final long MAX_BACKOFF_MS = 30_000;
    private static final long BACKOFF_MULTIPLIER = 2;

    private final OkHttpClient okHttpClient;
    private final SecureTokenStore tokenStore;
    private final AuthApiService authApi;
    private final WebSocketMessageParser parser;
    private final ExecutorService refreshExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "sip-token-refresh");
        thread.setDaemon(true);
        return thread;
    });
    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final AtomicBoolean stopped = new AtomicBoolean(true);
    private final AtomicBoolean reconnectScheduled = new AtomicBoolean(false);
    private final AtomicBoolean refreshInProgress = new AtomicBoolean(false);

    private volatile WebSocket webSocket;
    private volatile String currentBaseUrl;
    private volatile long backoffMs = MIN_BACKOFF_MS;

    @Inject
    public SipWebSocketClient(OkHttpClient okHttpClient, SecureTokenStore tokenStore,
                              AuthApiService authApi, WebSocketMessageParser parser) {
        this.okHttpClient = okHttpClient;
        this.tokenStore = tokenStore;
        this.authApi = authApi;
        this.parser = parser;
    }

    public void addListener(Listener listener) { if (listener != null) listeners.addIfAbsent(listener); }
    public void removeListener(Listener listener) { listeners.remove(listener); }

    public synchronized void connect(String baseUrl) {
        if (baseUrl == null || baseUrl.trim().isEmpty()) {
            emit(new WebSocketEvent.OnConnectionStateChanged(
                    WebSocketEvent.ConnectionState.FAILED, "Backend URL is empty"));
            return;
        }
        if (!stopped.get() && baseUrl.equals(currentBaseUrl) && webSocket != null) return;

        stopped.set(true);
        handler.removeCallbacksAndMessages(null);
        reconnectScheduled.set(false);
        WebSocket old = webSocket;
        webSocket = null;
        if (old != null) old.cancel();

        currentBaseUrl = baseUrl.trim();
        backoffMs = MIN_BACKOFF_MS;
        stopped.set(false);
        openSocket(currentBaseUrl);
    }

    public synchronized void disconnect() {
        stopped.set(true);
        currentBaseUrl = null;
        handler.removeCallbacksAndMessages(null);
        reconnectScheduled.set(false);
        WebSocket current = webSocket;
        webSocket = null;
        if (current != null) current.close(1000, "client disconnect");
        emit(new WebSocketEvent.OnConnectionStateChanged(
                WebSocketEvent.ConnectionState.DISCONNECTED, "manual"));
    }

    public boolean send(String json) {
        WebSocket current = webSocket;
        return current != null && current.send(json);
    }

    private void openSocket(String baseUrl) {
        if (stopped.get() || !baseUrl.equals(currentBaseUrl)) return;
        String token = tokenStore.getAccessToken();
        if (token == null) {
            refreshTokenThenReconnect();
            return;
        }

        String wsUrl = baseUrl.replaceFirst("^https://", "wss://")
                .replaceFirst("^http://", "ws://");
        if (!wsUrl.endsWith("/")) wsUrl += "/";
        wsUrl += "ws/events";

        Request request = new Request.Builder()
                .url(wsUrl)
                .header("Authorization", "Bearer " + token)
                .build();

        emit(new WebSocketEvent.OnConnectionStateChanged(
                WebSocketEvent.ConnectionState.CONNECTING, null));

        WebSocket created = okHttpClient.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket socket, Response response) {
                if (stopped.get() || !baseUrl.equals(currentBaseUrl) || webSocket != socket) {
                    socket.close(1000, "stale connection");
                    return;
                }
                backoffMs = MIN_BACKOFF_MS;
                reconnectScheduled.set(false);
                subscribe(socket, Set.of("incidents", "nodes"));
                emit(new WebSocketEvent.OnConnectionStateChanged(
                        WebSocketEvent.ConnectionState.CONNECTED, null));
            }

            @Override
            public void onMessage(WebSocket socket, String text) {
                if (socket != webSocket) return;
                WebSocketEvent event = parser.parse(text);
                if (event != null) emit(event);
            }

            @Override
            public void onMessage(WebSocket socket, ByteString bytes) {
                onMessage(socket, bytes.utf8());
            }

            @Override
            public void onFailure(WebSocket socket, Throwable t, Response response) {
                if (socket != webSocket) return;
                webSocket = null;
                emit(new WebSocketEvent.OnConnectionStateChanged(
                        WebSocketEvent.ConnectionState.FAILED,
                        t == null ? "WebSocket connection failed" : t.getMessage()));
                scheduleReconnect(baseUrl);
            }

            @Override
            public void onClosing(WebSocket socket, int code, String reason) {
                socket.close(code, reason);
            }

            @Override
            public void onClosed(WebSocket socket, int code, String reason) {
                if (socket != webSocket) return;
                webSocket = null;
                if (!stopped.get() && baseUrl.equals(currentBaseUrl)) {
                    emit(new WebSocketEvent.OnConnectionStateChanged(
                            WebSocketEvent.ConnectionState.DISCONNECTED, reason));
                    scheduleReconnect(baseUrl);
                }
            }
        });
        webSocket = created;
    }

    /**
     * A WebSocket handshake does not run the OkHttp HTTP authenticator. Refresh the access
     * token explicitly when it is expired, without blocking the main thread.
     */
    private void refreshTokenThenReconnect() {
        if (!refreshInProgress.compareAndSet(false, true)) return;
        String refreshToken = tokenStore.getRefreshToken();
        if (refreshToken == null || refreshToken.isBlank()) {
            refreshInProgress.set(false);
            emit(new WebSocketEvent.OnConnectionStateChanged(
                    WebSocketEvent.ConnectionState.FAILED, "Session expired. Please sign in again."));
            return;
        }

        refreshExecutor.execute(() -> {
            boolean retry = false;
            try {
                retrofit2.Response<LoginResponse> response =
                        authApi.refresh(Map.of("refreshToken", refreshToken)).execute();
                LoginResponse tokens = response.body();
                if (response.isSuccessful() && tokens != null
                        && tokens.token != null && tokens.refreshToken != null
                        && tokens.expiresIn > 0) {
                    tokenStore.save(tokens.token, tokens.refreshToken, tokens.expiresIn);
                    handler.post(() -> {
                        refreshInProgress.set(false);
                        String target = currentBaseUrl;
                        if (!stopped.get() && target != null) openSocket(target);
                    });
                    return;
                }
                retry = response.code() >= 500 || response.code() == 429;
                if (!retry) tokenStore.clear();
            } catch (IOException e) {
                retry = true;
            } catch (RuntimeException e) {
                retry = false;
                tokenStore.clear();
            }

            refreshInProgress.set(false);
            if (retry) {
                String target = currentBaseUrl;
                if (!stopped.get() && target != null) scheduleReconnect(target);
            } else {
                emit(new WebSocketEvent.OnConnectionStateChanged(
                        WebSocketEvent.ConnectionState.FAILED, "Session expired. Please sign in again."));
            }
        });
    }

    private void subscribe(WebSocket socket, Set<String> channels) {
        socket.send("{\"type\":\"SUBSCRIBE\",\"payload\":{\"channels\":["
                + joinQuoted(channels) + "]}}");
    }

    private static String joinQuoted(Set<String> values) {
        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (sb.length() > 0) sb.append(',');
            sb.append('"').append(value).append('"');
        }
        return sb.toString();
    }

    private void scheduleReconnect(String baseUrl) {
        if (stopped.get() || !baseUrl.equals(currentBaseUrl)
                || !reconnectScheduled.compareAndSet(false, true)) return;
        final long delay = backoffMs;
        backoffMs = Math.min(backoffMs * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS);
        handler.postDelayed(() -> {
            reconnectScheduled.set(false);
            if (!stopped.get() && baseUrl.equals(currentBaseUrl)) openSocket(baseUrl);
        }, TimeUnit.MILLISECONDS.toMillis(delay));
    }

    private void emit(WebSocketEvent event) {
        for (Listener listener : listeners) {
            try { listener.onEvent(event); }
            catch (RuntimeException ignored) { /* A listener must not break socket delivery. */ }
        }
    }
}
