package com.sip.guardian.di;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.room.Room;
import androidx.room.migration.Migration;
import androidx.sqlite.db.SupportSQLiteDatabase;

import com.sip.guardian.data.local.SipDatabase;
import com.sip.guardian.data.local.dao.IncidentDao;
import com.sip.guardian.data.local.dao.NodeDao;

import javax.inject.Singleton;

import dagger.Module;
import dagger.Provides;
import dagger.hilt.InstallIn;
import dagger.hilt.android.qualifiers.ApplicationContext;
import dagger.hilt.components.SingletonComponent;

@Module
@InstallIn(SingletonComponent.class)
public class AppModule {

    private static final Migration MIGRATION_1_2 = new Migration(1, 2) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase database) {
            database.execSQL("ALTER TABLE incidents ADD COLUMN responseEventsJson TEXT");
        }
    };

    @Provides
    @Singleton
    SipDatabase provideDatabase(@ApplicationContext Context context) {
        return Room.databaseBuilder(context, SipDatabase.class, "sip.db")
                .addMigrations(MIGRATION_1_2)
                .build();
    }

    @Provides
    IncidentDao provideIncidentDao(SipDatabase db) { return db.incidentDao(); }

    @Provides
    NodeDao provideNodeDao(SipDatabase db) { return db.nodeDao(); }
}
