package com.sip.guardian.ui.screen.incidents

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sip.guardian.domain.model.HumanLabel
import com.sip.guardian.domain.model.Incident
import com.sip.guardian.domain.repository.IncidentRepository
import com.sip.guardian.domain.usecase.VerifyIncidentUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class IncidentDetailUiState(
    val loading: Boolean = true,
    val incident: Incident? = null,
    val error: String? = null,
    val submitting: Boolean = false,
    val completed: Boolean = false,
)

@HiltViewModel
class IncidentDetailViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val repository: IncidentRepository,
    private val verifyIncident: VerifyIncidentUseCase,
) : ViewModel() {
    private val incidentId: String = checkNotNull(savedStateHandle["incidentId"])
    private val _uiState = MutableStateFlow(IncidentDetailUiState())
    val uiState: StateFlow<IncidentDetailUiState> = _uiState

    init { refresh() }

    fun refresh() {
        _uiState.value = _uiState.value.copy(loading = true, error = null)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val incident = repository.getIncidentById(incidentId)
                _uiState.value = if (incident == null) {
                    IncidentDetailUiState(loading = false, error = "Incident not found")
                } else {
                    IncidentDetailUiState(loading = false, incident = incident)
                }
            } catch (e: Exception) {
                _uiState.value = IncidentDetailUiState(
                    loading = false,
                    error = e.message ?: "Unable to load incident",
                )
            }
        }
    }

    fun verify(label: HumanLabel, notes: String) {
        _uiState.value = _uiState.value.copy(submitting = true, error = null)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val result = verifyIncident.execute(incidentId, label, notes)
                _uiState.value = if (result.isSuccess) {
                    IncidentDetailUiState(
                        loading = false,
                        incident = result.incident,
                        completed = true,
                    )
                } else {
                    _uiState.value.copy(
                        submitting = false,
                        error = result.error ?: "Verification failed",
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    submitting = false,
                    error = e.message ?: "Verification failed",
                )
            }
        }
    }
}
