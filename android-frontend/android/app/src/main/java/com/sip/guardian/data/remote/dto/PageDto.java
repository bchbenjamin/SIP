package com.sip.guardian.data.remote.dto;

import java.util.List;

public class PageDto<T> {
    public List<T> content;
    public int page;
    public int size;
    public long totalElements;
    public int totalPages;
}
