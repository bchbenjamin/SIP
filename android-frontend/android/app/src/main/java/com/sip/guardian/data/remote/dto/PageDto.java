package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class PageDto<T> {
    @SerializedName("content")
    public List<T> content;

    @SerializedName("page")
    public int page;

    @SerializedName("size")
    public int size;

    @SerializedName("totalElements")
    public long totalElements;

    @SerializedName("totalPages")
    public int totalPages;
}
