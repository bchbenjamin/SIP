package com.sip.guardian.domain.model;

/** Authenticated operator. Role model kept minimal but extensible (plan §26). */
public final class User {
    public enum Role { ADMIN, OPERATOR }

    private final String id;
    private final String username;
    private final Role role;

    public User(String id, String username, Role role) {
        this.id = id;
        this.username = username;
        this.role = role;
    }

    public String getId() { return id; }
    public String getUsername() { return username; }
    public Role getRole() { return role; }

    public boolean canVerify() { return role == Role.OPERATOR || role == Role.ADMIN; }
    public boolean canManagePolicy() { return role == Role.ADMIN; }
}
