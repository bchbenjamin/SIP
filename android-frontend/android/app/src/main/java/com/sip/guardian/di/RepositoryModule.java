package com.sip.guardian.di;

import com.sip.guardian.data.repository.AuthRepositoryImpl;
import com.sip.guardian.data.repository.AutopilotRepositoryImpl;
import com.sip.guardian.data.repository.DatasetRepositoryImpl;
import com.sip.guardian.data.repository.EvidenceRepositoryImpl;
import com.sip.guardian.data.repository.IncidentRepositoryImpl;
import com.sip.guardian.data.repository.NodeRepositoryImpl;
import com.sip.guardian.domain.repository.AuthRepository;
import com.sip.guardian.domain.repository.AutopilotRepository;
import com.sip.guardian.domain.repository.DatasetRepository;
import com.sip.guardian.domain.repository.EvidenceRepository;
import com.sip.guardian.domain.repository.IncidentRepository;
import com.sip.guardian.domain.repository.NodeRepository;

import dagger.Binds;
import dagger.Module;
import dagger.hilt.InstallIn;
import dagger.hilt.components.SingletonComponent;

@Module
@InstallIn(SingletonComponent.class)
public abstract class RepositoryModule {

    @Binds
    abstract IncidentRepository bindIncidentRepository(IncidentRepositoryImpl impl);

    @Binds
    abstract NodeRepository bindNodeRepository(NodeRepositoryImpl impl);

    @Binds
    abstract AutopilotRepository bindAutopilotRepository(AutopilotRepositoryImpl impl);

    @Binds
    abstract EvidenceRepository bindEvidenceRepository(EvidenceRepositoryImpl impl);

    @Binds
    abstract AuthRepository bindAuthRepository(AuthRepositoryImpl impl);

    @Binds
    abstract DatasetRepository bindDatasetRepository(DatasetRepositoryImpl impl);
}
