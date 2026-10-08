package com.sip.guardian.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.sip.guardian.ui.theme.SipCritical
import com.sip.guardian.ui.theme.SipSafe
import com.sip.guardian.ui.theme.SipWarning

/** Gradient-style confidence indicator: red -> yellow -> green (plan §22). */
@Composable
fun ConfidenceMeter(confidence: Float, modifier: Modifier = Modifier) {
    val color: Color = when {
        confidence < 0.5f -> SipCritical
        confidence < 0.8f -> SipWarning
        else -> SipSafe
    }
    Box(modifier = modifier) {
        LinearProgressIndicator(
            progress = { confidence.coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            color = color,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
        )
    }
    Text(
        text = "${(confidence * 100).toInt()}%",
        style = MaterialTheme.typography.labelSmall,
        color = color,
    )
}
