package com.sip.guardian.ui.screen.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sip.guardian.data.remote.api.SipApiService
import com.sip.guardian.data.remote.dto.DashboardDto
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class DashboardUiState(
    val loading: Boolean = true,
    val totalNodes: Int = 0,
    val onlineNodes: Int = 0,
    val activeThreats: Int = 0,
    val systemHealth: String = "UNKNOWN",
    val error: String? = null,
)

@HiltViewModel
class DashboardViewModel @Inject constructor(
    private val api: SipApiService,
) : ViewModel() {
    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState

    fun refresh() {
        _uiState.value = _uiState.value.copy(loading = true, error = null)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val r = api.getDashboard().execute()
                val body: DashboardDto? = if (r.isSuccessful) r.body() else null
                if (body != null) {
                    _uiState.value = DashboardUiState(
                        loading = false,
                        totalNodes = body.totalNodes,
                        onlineNodes = body.onlineNodes,
                        activeThreats = body.activeThreats,
                        systemHealth = body.systemHealth,
                    )
                } else {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = "Backend unavailable",
                    )
                }
            } catch (_: Exception) {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    error = "Backend unavailable",
                )
            }
        }
    }
}
