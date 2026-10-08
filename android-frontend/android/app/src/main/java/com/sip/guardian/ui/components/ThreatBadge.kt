package com.sip.guardian.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sip.guardian.domain.model.ThreatType
import com.sip.guardian.ui.theme.SipCritical
import com.sip.guardian.ui.theme.SipInfo
import com.sip.guardian.ui.theme.SipMuted
import com.sip.guardian.ui.theme.SipWarning

/** Color-coded threat type badge (plan §21/§22). */
@Composable
fun ThreatBadge(threatType: ThreatType, modifier: Modifier = Modifier) {
    val (label, color) = when (threatType) {
        ThreatType.WEAPON -> "WEAPON" to SipCritical
        ThreatType.ASSAULT -> "ASSAULT" to SipCritical
        ThreatType.TRESPASS -> "TRESPASS" to SipWarning
        ThreatType.LOITERING -> "LOITERING" to SipWarning
        ThreatType.WILDLIFE -> "WILDLIFE" to SipInfo
        ThreatType.SUSPICIOUS_OBJECT -> "SUSPICIOUS" to SipWarning
        ThreatType.UNKNOWN -> "UNKNOWN" to SipMuted
    }
    Box(
        modifier = modifier
            .background(color.copy(alpha = 0.18f), RoundedCornerShape(6.dp))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text(
            text = label,
            color = color,
            fontSize = 11.sp,
            style = MaterialTheme.typography.labelSmall,
        )
    }
}
