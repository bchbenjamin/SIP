package com.sip.guardian.ui.navigation

sealed class SipNavDestination(val route: String) {
    data object Login : SipNavDestination("login")
    data object Dashboard : SipNavDestination("dashboard")
    data object IncidentFeed : SipNavDestination("incidents")
    data object NodeMap : SipNavDestination("nodes")
    data object Autopilot : SipNavDestination("autopilot")
    data object More : SipNavDestination("more")
    data object EventHistory : SipNavDestination("history")
    data object Settings : SipNavDestination("settings")

    data object IncidentDetail : SipNavDestination("incidents/{incidentId}") {
        const val ARG_ID = "incidentId"
        fun createRoute(id: String) = "incidents/$id"
    }
}
