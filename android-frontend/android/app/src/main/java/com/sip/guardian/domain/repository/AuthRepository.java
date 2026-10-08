package com.sip.guardian.domain.repository;

import com.sip.guardian.domain.model.User;

public interface AuthRepository {
    User login(String username, String password);
    /** Refresh the access token; throws if the session is dead. */
    User refreshSession();
    void logout();
    User currentUser();
}
