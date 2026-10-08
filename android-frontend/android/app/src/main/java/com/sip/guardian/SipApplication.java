package com.sip.guardian;

import android.app.Application;

import dagger.hilt.android.HiltAndroidApp;

@HiltAndroidApp
public class SipApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        // Hilt wires DI. StrictMode/Timber hooks can be added for debug builds.
    }
}
