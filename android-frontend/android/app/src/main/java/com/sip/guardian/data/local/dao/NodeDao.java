package com.sip.guardian.data.local.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import com.sip.guardian.data.local.entity.NodeEntity;

import java.util.List;

@Dao
public interface NodeDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void upsertAll(List<NodeEntity> nodes);

    @Query("SELECT * FROM nodes")
    List<NodeEntity> getAll();
}
