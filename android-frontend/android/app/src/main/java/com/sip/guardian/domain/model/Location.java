package com.sip.guardian.domain.model;

/** Geographic location of an incident or node. */
public final class Location {
    private final double latitude;
    private final double longitude;
    private final String humanReadable;
    private final double accuracyMeters;

    public Location(double latitude, double longitude, String humanReadable, double accuracyMeters) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.humanReadable = humanReadable == null ? "" : humanReadable;
        this.accuracyMeters = accuracyMeters;
    }

    public double getLatitude() { return latitude; }
    public double getLongitude() { return longitude; }
    public String getHumanReadable() { return humanReadable; }
    public double getAccuracyMeters() { return accuracyMeters; }
}
