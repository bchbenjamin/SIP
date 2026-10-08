package com.sip.guardian.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val DarkColors = darkColorScheme(
    primary = SipPrimary,
    secondary = SipSecondary,
    tertiary = SipTertiary,
    background = SipDarkBackground,
    surface = SipDarkSurface,
    surfaceVariant = SipDarkSurfaceVariant,
    error = SipCritical,
    onPrimary = SipDarkBackground,
    onSecondary = SipDarkBackground,
    onBackground = Color(0xFFC9D1D9),
    onSurface = Color(0xFFC9D1D9),
    onSurfaceVariant = SipMuted,
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF0969DA),
    secondary = SipSecondary,
    tertiary = SipTertiary,
    error = SipCritical,
)

private val SipShapes = Shapes(
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(16.dp), // plan §22: cards 16dp
)

@Composable
fun SIPGuardianTheme(
    darkTheme: Boolean = true, // operations aesthetic defaults to dark
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        shapes = SipShapes,
        content = content,
    )
}
