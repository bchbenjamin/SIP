package com.sip.guardian.service;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

import androidx.core.app.NotificationCompat;

import com.sip.guardian.R;
import com.sip.guardian.ui.MainActivity;

import java.util.concurrent.atomic.AtomicInteger;

public final class NotificationService {

    private static final String CHANNEL_ID = "sip_events";
    private static final String CHANNEL_ALERTS = "sip_alerts";
    public static final String EXTRA_INCIDENT_ID = "com.sip.guardian.EXTRA_INCIDENT_ID";
    private static final AtomicInteger NEXT_ID = new AtomicInteger(100);

    private NotificationService() {}

    public static void notifyNewIncident(Context context, String incidentId,
                                         String threatType, String description) {
        ensureChannels(context);

        Intent open = new Intent(context, MainActivity.class)
                .setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP)
                .putExtra(EXTRA_INCIDENT_ID, incidentId);
        PendingIntent pi = PendingIntent.getActivity(
                context,
                incidentId == null ? NEXT_ID.get() : incidentId.hashCode(),
                open,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );

        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle("Threat detected: " + (threatType == null ? "UNKNOWN" : threatType))
                .setContentText(description == null ? "" : description)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setContentIntent(pi);

        notify(context, NEXT_ID.getAndIncrement(), builder);
    }

    public static void notifySystemAlert(Context context, String level, String message) {
        ensureChannels(context);

        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ALERTS)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle("SIP " + (level == null ? "ALERT" : level))
                .setContentText(message == null ? "" : message)
                .setAutoCancel(true);

        notify(context, NEXT_ID.getAndIncrement(), builder);
    }

    private static void notify(Context context, int id, NotificationCompat.Builder builder) {
        NotificationManager nm = manager(context);
        if (nm != null) {
            nm.notify(id, builder.build());
        }
    }

    private static NotificationManager manager(Context context) {
        return context.getSystemService(NotificationManager.class);
    }

    private static void ensureChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationManager nm = manager(context);
        if (nm == null) return;

        nm.createNotificationChannel(new NotificationChannel(
                CHANNEL_ID, "Incidents", NotificationManager.IMPORTANCE_HIGH));
        nm.createNotificationChannel(new NotificationChannel(
                CHANNEL_ALERTS, "System alerts", NotificationManager.IMPORTANCE_DEFAULT));
    }
}
