package com.sip.guardian.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.navigation.compose.rememberNavController
import com.sip.guardian.BuildConfig
import com.sip.guardian.domain.repository.AuthRepository
import com.sip.guardian.service.NotificationService
import com.sip.guardian.service.WebSocketService
import com.sip.guardian.ui.navigation.SipNavDestination
import com.sip.guardian.ui.navigation.SipNavGraph
import com.sip.guardian.ui.theme.SIPGuardianTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    @Inject lateinit var authRepository: AuthRepository

    private var pendingIncidentId by mutableStateOf<String?>(null)

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            startRealtimeMonitoring()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val authenticated = authRepository.currentUser() != null
        val start = if (authenticated) {
            SipNavDestination.Dashboard.route
        } else {
            SipNavDestination.Login.route
        }

        setContent {
            SIPGuardianTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    val navController = rememberNavController()

                    LaunchedEffect(authenticated) {
                        if (authenticated) {
                            requestNotificationPermissionAndStartMonitoring()
                        }
                    }
                    LaunchedEffect(authenticated, pendingIncidentId) {
                        if (authenticated) {
                            pendingIncidentId?.let { id ->
                                navController.navigate(SipNavDestination.IncidentDetail.createRoute(id)) {
                                    launchSingleTop = true
                                }
                                pendingIncidentId = null
                            }
                        }
                    }

                    SipNavGraph(
                        navController = navController,
                        startDestination = start,
                        onAuthenticated = {
                            requestNotificationPermissionAndStartMonitoring()
                            pendingIncidentId?.let { id ->
                                navController.navigate(SipNavDestination.IncidentDetail.createRoute(id)) {
                                    launchSingleTop = true
                                }
                                pendingIncidentId = null
                            }
                        },
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        pendingIncidentId = intent.getStringExtra(NotificationService.EXTRA_INCIDENT_ID)
    }

    private fun requestNotificationPermissionAndStartMonitoring() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            startRealtimeMonitoring()
        }
    }

    private fun startRealtimeMonitoring() {
        val intent = Intent(this, WebSocketService::class.java)
            .putExtra(WebSocketService.EXTRA_BASE_URL, BuildConfig.API_BASE_URL)
        ContextCompat.startForegroundService(this, intent)
    }
}
