package com.sip.guardian.ui.theme

import androidx.compose.ui.graphics.Color

// SIP Guardian Theme — Security/Operations aesthetic (plan §22)
val SipDarkBackground = Color(0xFF0D1117)
val SipDarkSurface = Color(0xFF161B22)
val SipDarkSurfaceVariant = Color(0xFF21262D)
val SipPrimary = Color(0xFF58A6FF)
val SipSecondary = Color(0xFF3FB950)
val SipTertiary = Color(0xFFD2A8FF)

// Severity colors
val SipCritical = Color(0xFFF85149)
val SipWarning = Color(0xFFD29922)
val SipSafe = Color(0xFF3FB950)
val SipInfo = Color(0xFF58A6FF)
val SipMuted = Color(0xFF8B949E)

val NodeOnline = SipSafe
val NodeDegraded = SipWarning
val NodeOffline = SipCritical

val ThreatLow = Color(0xFF388BFD)
val ThreatMedium = SipWarning
val ThreatHigh = Color(0xFFDA3633)
val ThreatCritical = Color(0xFFF85149)
