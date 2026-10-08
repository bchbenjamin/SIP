package com.sip.guardian.ui.screen.incidents

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.sip.guardian.domain.model.HumanLabel
import com.sip.guardian.ui.components.ConfidenceMeter
import com.sip.guardian.ui.components.ThreatBadge

@Composable
fun IncidentDetailScreen(
    onBack: () -> Unit,
    viewModel: IncidentDetailViewModel = hiltViewModel(),
) {
    val state by viewModel.uiState.collectAsState()
    var notes by remember { mutableStateOf("") }

    val incident = state.incident

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedButton(onClick = onBack) { Text("Back") }

        when {
            state.loading -> CircularProgressIndicator()
            incident == null -> Text(
                state.error ?: "Incident unavailable",
                color = MaterialTheme.colorScheme.error,
            )
            else -> {
                ThreatBadge(incident.threat.type)
                Text(
                    incident.threat.description.ifEmpty { "Incident " + incident.id },
                    style = MaterialTheme.typography.headlineSmall,
                )
                Text("State: " + incident.state)
                Text("Node: " + incident.nodeId)
                incident.detectionResult?.let {
                    Text("Detection: " + it.predictedClass)
                    ConfidenceMeter(it.confidence.toFloat())
                }
                HorizontalDivider()

                if (incident.requiresVerification() && !state.completed) {
                    Text("Operator verification",
                        style = MaterialTheme.typography.titleMedium)
                    OutlinedTextField(
                        value = notes,
                        onValueChange = { notes = it },
                        label = { Text("Notes (optional)") },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            enabled = !state.submitting,
                            onClick = { viewModel.verify(HumanLabel.TRUE_POSITIVE, notes) },
                        ) { Text("Verify") }
                        OutlinedButton(
                            enabled = !state.submitting,
                            onClick = { viewModel.verify(HumanLabel.FALSE_POSITIVE, notes) },
                        ) { Text("Reject") }
                        OutlinedButton(
                            enabled = !state.submitting,
                            onClick = { viewModel.verify(HumanLabel.UNCERTAIN, notes) },
                        ) { Text("Uncertain") }
                    }
                } else if (state.completed) {
                    Text("Verification submitted.",
                        color = MaterialTheme.colorScheme.primary)
                }

                state.error?.let {
                    Text(it, color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}
