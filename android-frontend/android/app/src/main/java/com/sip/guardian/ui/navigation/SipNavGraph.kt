package com.sip.guardian.ui.navigation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.navArgument
import com.sip.guardian.ui.screen.auth.LoginScreen
import com.sip.guardian.ui.screen.dashboard.DashboardScreen
import com.sip.guardian.ui.screen.incidents.IncidentDetailScreen
import com.sip.guardian.ui.screen.incidents.IncidentFeedScreen

private val topLevel = listOf(
    SipNavDestination.Dashboard,
    SipNavDestination.IncidentFeed,
)

@Composable
fun SipNavGraph(
    navController: NavHostController,
    startDestination: String,
    onAuthenticated: () -> Unit = {},
) {
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route
    val showNavigation = currentRoute != SipNavDestination.Login.route &&
            currentRoute != SipNavDestination.IncidentDetail.route

    Scaffold(
        bottomBar = {
            if (showNavigation) {
                NavigationBar {
                    topLevel.forEach { destination ->
                        NavigationBarItem(
                            selected = currentRoute == destination.route,
                            onClick = {
                                navController.navigate(destination.route) {
                                    launchSingleTop = true
                                    restoreState = true
                                    popUpTo(SipNavDestination.Dashboard.route) {
                                        saveState = true
                                    }
                                }
                            },
                            icon = { Text(if (destination == SipNavDestination.Dashboard) "⌂" else "!") },
                            label = {
                                Text(if (destination == SipNavDestination.Dashboard) "Dashboard" else "Incidents")
                            },
                        )
                    }
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = startDestination,
        ) {
            composable(SipNavDestination.Login.route) {
                LoginScreen(onLoggedIn = {
                    navController.navigate(SipNavDestination.Dashboard.route) {
                        popUpTo(SipNavDestination.Login.route) { inclusive = true }
                    }
                    onAuthenticated()
                })
            }
            composable(SipNavDestination.Dashboard.route) {
                Box(modifier = Modifier.padding(padding)) {
                    DashboardScreen()
                }
            }
            composable(SipNavDestination.IncidentFeed.route) {
                Box(modifier = Modifier.padding(padding)) {
                    IncidentFeedScreen(onIncidentClick = { id ->
                        navController.navigate(SipNavDestination.IncidentDetail.createRoute(id))
                    })
                }
            }
            composable(
                route = SipNavDestination.IncidentDetail.route,
                arguments = listOf(navArgument(SipNavDestination.IncidentDetail.ARG_ID) {
                    type = NavType.StringType
                }),
            ) {
                IncidentDetailScreen(onBack = { navController.popBackStack() })
            }
        }
    }
}
