package com.sip.guardian.data.remote.dto;

public class VerificationRequest {
    public String verdict;   // VERIFIED | REJECTED
    public String label;     // TRUE_POSITIVE | FALSE_POSITIVE | UNCERTAIN
    public String notes;

    public VerificationRequest(String verdict, String label, String notes) {
        this.verdict = verdict;
        this.label = label;
        this.notes = notes;
    }
}
