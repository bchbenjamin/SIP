package com.sip.guardian.data.local;

import androidx.room.Database;
import androidx.room.RoomDatabase;

import com.sip.guardian.data.local.dao.IncidentDao;
import com.sip.guardian.data.local.dao.NodeDao;
import com.sip.guardian.data.local.entity.IncidentEntity;
import com.sip.guardian.data.local.entity.NodeEntity;

@Database(entities = {IncidentEntity.class, NodeEntity.class},
          version = 2, exportSchema = false)
public abstract class SipDatabase extends RoomDatabase {
    public abstract IncidentDao incidentDao();
    public abstract NodeDao nodeDao();
}
