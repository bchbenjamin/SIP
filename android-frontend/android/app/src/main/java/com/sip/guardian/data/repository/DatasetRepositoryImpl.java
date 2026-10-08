package com.sip.guardian.data.repository;

import android.content.Context;

import com.sip.guardian.data.remote.api.SipApiService;
import com.sip.guardian.domain.repository.DatasetRepository;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.Map;

import dagger.hilt.android.qualifiers.ApplicationContext;
import javax.inject.Inject;
import javax.inject.Singleton;

import okhttp3.ResponseBody;
import retrofit2.Response;

@Singleton
public class DatasetRepositoryImpl implements DatasetRepository {

    private final SipApiService api;
    private final File exportDir;

    @Inject
    public DatasetRepositoryImpl(SipApiService api, @ApplicationContext Context context) {
        this.api = api;
        this.exportDir = context.getExternalFilesDir("exports");
        if (exportDir != null) exportDir.mkdirs();
    }

    @Override
    public File export(String format, Instant from, Instant to, String labels) {
        try {
            Response<ResponseBody> r = api.exportDataset(format,
                    from != null ? from.toString() : null,
                    to != null ? to.toString() : null, labels).execute();
            if (!r.isSuccessful() || r.body() == null) {
                throw new IllegalStateException("Export failed: HTTP " + r.code());
            }
            File zip = new File(exportDir,
                    "dataset_" + format + "_" + System.currentTimeMillis() + ".zip");
            try (InputStream in = r.body().byteStream();
                 FileOutputStream out = new FileOutputStream(zip)) {
                byte[] buf = new byte[8192];
                int read;
                while ((read = in.read(buf)) != -1) out.write(buf, 0, read);
            }
            return zip;
        } catch (IOException e) {
            throw new IllegalStateException("Export failed: network error", e);
        }
    }

    @Override
    public Map<String, Object> stats() {
        try {
            Response<Map<String, Object>> r = api.getDatasetStats().execute();
            if (r.isSuccessful() && r.body() != null) return r.body();
        } catch (IOException ignored) { }
        return Map.of("totalLabeled", 0, "truePositives", 0, "falsePositives", 0);
    }
}
