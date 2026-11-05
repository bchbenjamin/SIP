# SIP Guardian application-specific R8 rules
-keepattributes Signature, InnerClasses, EnclosingMethod, *Annotation*
-keep class com.sip.guardian.domain.model.** { *; }
-keep class com.sip.guardian.data.remote.dto.** { *; }
-keep interface com.sip.guardian.data.remote.api.** { *; }
