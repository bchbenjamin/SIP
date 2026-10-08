package com.sip.guardian.ui.screen.incidents

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.sip.guardian.domain.model.IncidentState
import com.sip.guardian.ui.components.IncidentCard
import com.sip.guardian.ui.theme.SipWarning

/** Chronological incident list with filters (plan §21). */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IncidentFeedScreen(
    onIncidentClick: (String) -> Unit,
    viewModel: IncidentFeedViewModel = hiltViewModel(),
) {
    val state by viewModel.uiState.collectAsState()

    Column(Modifier.fillMaxSize()) {
        // Filter chips
        androidx.compose.foundation.lazy.LazyRow(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp),
        ) {
            item {
                FilterChip(
                    selected = state.stateFilter == null,
                    onClick = { viewModel.setStateFilter(null) },
                    label = { Text("All states") },
                )
            }
            items(IncidentState.entries.filter {
                it == IncidentState.PENDING_VERIFICATION || it == IncidentState.VERIFIED
                        || it == IncidentState.REJECTED || it == IncidentState.ESCALATED
            }) { s ->
                FilterChip(
                    selected = state.stateFilter == s,
                    onClick = { viewModel.setStateFilter(if (state.stateFilter == s) null else s) },
                    label = { Text(s.name) },
                )
            }
        }

        if (state.offline) {
            Text(
                "Offline — showing cached incidents",
                color = SipWarning,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        LazyColumn(Modifier.fillMaxSize()) {
            items(state.incidents, key = { it.id }) { incident ->
                IncidentCard(incident = incident, onClick = { onIncidentClick(incident.id) })
            }
        }
    }
}
