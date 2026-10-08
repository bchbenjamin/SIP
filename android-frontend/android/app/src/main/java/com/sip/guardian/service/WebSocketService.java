package com.sip.guardian.service;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;

import androidx.core.app.NotificationCompat;
import androidx.core.app.ServiceCompat;

import com.sip.guardian.BuildConfig;
import com.sip.guardian.R;
import com.sip.guardian.data.local.dao.IncidentDao;
import com.sip.guardian.data.mapper.IncidentMapper;
import com.sip.guardian.data.remote.websocket.SipWebSocketClient;
import com.sip.guardian.data.remote.websocket.WebSocketEvent;

import javax.inject.Inject;

import dagger.hilt.android.AndroidEntryPoint;

/**
 * Foreground service holding the authenticated WebSocket while the app is active
 * and for a bounded period after backgrounding. Missed events are reconciled via
 * REST when the app resumes.
 */
@AndroidEntryPoint
public class WebSocketService extends Service {

    public static final String CHANNEL_ID = "sip_events";
    private static final int NOTIFICATION_ID = 1;
    public static final String EXTRA_BASE_URL = "base_url";

    @Inject SipWebSocketClient webSocketClient;
    @Inject IncidentDao incidentDao;
    @Inject IncidentMapper incidentMapper;

    private final SipWebSocketClient.Listener listener = new SipWebSocketClient.Listener() {
        @Override
        public void onEvent(WebSocketEvent event) {
            if (event instanceof WebSocketEvent.OnIncidentReceived) {
                WebSocketEvent.OnIncidentReceived e = (WebSocketEvent.OnIncidentReceived) event;
                if (e.incident == null || e.incident.id == null) return;
                incidentDao.upsert(incidentMapper.toEntity(e.incident));
                String threatType = e.incident.threat != null ? e.incident.threat.type
                        : e.incident.threatType != null ? e.incident.threatType : "UNKNOWN";
                String description = e.incident.threat != null ? e.incident.threat.description
                        : e.incident.threatDescription != null ? e.incident.threatDescription : "";
                NotificationService.notifyNewIncident(WebSocketService.this,
                        e.incident.id, threatType, description);
            } else if (event instanceof WebSocketEvent.OnIncidentUpdated) {
                WebSocketEvent.OnIncidentUpdated e = (WebSocketEvent.OnIncidentUpdated) event;
                // The backend sends the full canonical DTO; update Room so observers see the change.
                if (e.incident != null && e.incident.id != null) {
                    incidentDao.upsert(incidentMapper.toEntity(e.incident));
                }
            } else if (event instanceof WebSocketEvent.OnSystemAlert) {
                WebSocketEvent.OnSystemAlert e = (WebSocketEvent.OnSystemAlert) event;
                NotificationService.notifySystemAlert(WebSocketService.this,
                        e.level, e.message);
            }
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
        webSocketClient.addListener(listener);

        Notification notification = buildForegroundNotification("Monitoring…");
        int foregroundType = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                ? ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                : 0;
        ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                notification,
                foregroundType
        );
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String baseUrl = intent != null ? intent.getStringExtra(EXTRA_BASE_URL) : null;
        if (baseUrl == null || baseUrl.trim().isEmpty()) {
            baseUrl = BuildConfig.API_BASE_URL;
        }
        if (baseUrl != null && !baseUrl.trim().isEmpty()) {
            webSocketClient.connect(baseUrl);
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        webSocketClient.removeListener(listener);
        webSocketClient.disconnect();
        super.onDestroy();
    }

    @Override
    public void onTimeout(int startId, int fgsType) {
        // Android 15 limits dataSync foreground-service runtime.
        stopSelf(startId);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private Notification buildForegroundNotification(String text) {
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle("SIP Guardian")
                .setContentText(text)
                .setOngoing(true)
                .setCategory(NotificationCompat.CATEGORY_SERVICE)
                .build();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) {
            nm.createNotificationChannel(new NotificationChannel(
                    CHANNEL_ID,
                    "Incidents",
                    NotificationManager.IMPORTANCE_HIGH
            ));
        }
    }
}
