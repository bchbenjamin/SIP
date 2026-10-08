package com.sip.guardian.data.repository;

import android.content.Context;

import com.sip.guardian.data.remote.api.SipApiService;
import com.sip.guardian.domain.model.EvidenceBundle;
import com.sip.guardian.domain.model.EvidenceType;
import com.sip.guardian.domain.repository.EvidenceRepository;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import dagger.hilt.android.qualifiers.ApplicationContext;
import javax.inject.Inject;
import javax.inject.Singleton;
import okhttp3.ResponseBody;
import retrofit2.Response;

/**
 * Downloads evidence via authenticated GET into the app cache dir.
 * Coil handles images in the UI layer with its own 50MB disk cache;
 * this covers video/audio prefetch (plan §16).
 */
@Singleton
public class EvidenceRepositoryImpl implements EvidenceRepository {

    private static final long MAX_EVIDENCE_BYTES = 64L * 1024 * 1024;

    private final SipApiService api;
    private final File cacheDir;

    @Inject
    public EvidenceRepositoryImpl(SipApiService api, @ApplicationContext Context context) {
        this.api = api;
        this.cacheDir = new File(context.getCacheDir(), "evidence");
        //noinspection ResultOfMethodCallIgnored
        this.cacheDir.mkdirs();
    }

    @Override
    public File getEvidenceFile(String incidentId, EvidenceBundle bundle, EvidenceType type) {
        File target = new File(cacheDir, incidentId + "_" + type.name().toLowerCase());
        if (target.exists() && target.length() > 0) return target;

        try {
            Response<ResponseBody> response;
            switch (type) {
                case IMAGE:   response = api.getEvidenceImage(incidentId).execute(); break;
                case VIDEO:   response = api.getEvidenceVideo(incidentId).execute(); break;
                case AUDIO:   response = api.getEvidenceAudio(incidentId).execute(); break;
                default: throw new IllegalArgumentException("Unknown type " + type);
            }
            if (!response.isSuccessful() || response.body() == null) {
                throw new IOException("Evidence download failed: HTTP " + response.code());
            }
            writeToFile(response.body(), target);
            return target;
        } catch (IOException e) {
            //noinspection ResultOfMethodCallIgnored
            target.delete();
            throw new IllegalStateException("Evidence unavailable", e);
        }
    }

    private void writeToFile(ResponseBody body, File target) throws IOException {
        long total = 0;
        try (InputStream in = body.byteStream();
             FileOutputStream out = new FileOutputStream(target)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                total += read;
                if (total > MAX_EVIDENCE_BYTES) {
                    throw new IOException("Evidence exceeds size limit");
                }
                out.write(buffer, 0, read);
            }
        }
    }
}
