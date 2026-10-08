package com.sip.guardian.data.local.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import com.sip.guardian.data.local.entity.IncidentEntity;

import java.util.List;

import io.reactivex.rxjava3.core.Flowable;

@Dao
public interface IncidentDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void upsertAll(List<IncidentEntity> incidents);

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void upsert(IncidentEntity incident);

    @Query("SELECT * FROM incidents ORDER BY createdAtEpochMs DESC")
    Flowable<List<IncidentEntity>> observeAll();

    @Query("SELECT * FROM incidents WHERE id = :id")
    Flowable<IncidentEntity> observeById(String id);

    @Query("SELECT * FROM incidents WHERE id = :id")
    IncidentEntity getById(String id);

    @Query("SELECT * FROM incidents " +
            "WHERE (:state IS NULL OR state = :state) " +
            "AND (:threatType IS NULL OR threatType = :threatType) " +
            "AND (:nodeId IS NULL OR nodeId = :nodeId) " +
            "AND (:fromEpochMs IS NULL OR createdAtEpochMs >= :fromEpochMs) " +
            "AND (:toEpochMs IS NULL OR createdAtEpochMs <= :toEpochMs) " +
            "ORDER BY createdAtEpochMs DESC LIMIT :limit OFFSET :offset")
    List<IncidentEntity> getCachedFiltered(String state, String threatType, String nodeId,
                                           Long fromEpochMs, Long toEpochMs, int limit, int offset);

    @Query("SELECT * FROM incidents ORDER BY createdAtEpochMs DESC LIMIT :limit")
    List<IncidentEntity> getRecent(int limit);

    @Query("DELETE FROM incidents WHERE createdAtEpochMs < :cutoffEpochMs")
    int pruneOlderThan(long cutoffEpochMs);
}
