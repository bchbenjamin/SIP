package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;

public class VerificationRequest {
    @SerializedName("verdict")
    public String verdict;   // VERIFIED | REJECTED

    @SerializedName("label")
    public String label;     // TRUE_POSITIVE | FALSE_POSITIVE | UNCERTAIN

    @SerializedName("notes")
    public String notes;

    public VerificationRequest(String verdict, String label, String notes) {
        this.verdict = verdict;
        this.label = label;
        this.notes = notes;
    }
}
