import pandas as pd
import numba
import umap
import numpy as np

def ts_windowing(df, window_size, wide_columns, wide_tgt):
    tmp_df = df
    wide_df = pd.DataFrame()

    for idx in range(tmp_df.shape[0] - window_size):
        tmp_dt = {'anchor': tmp_df.iloc[-1, :][wide_tgt]}
        for row_idx in range(window_size):
            tmp_dict = {v + '_' + str(row_idx+1): tmp_df.iloc[row_idx*-1-2, :][v] for k, v in enumerate(wide_columns)}
            tmp_dt.update(tmp_dict)

        wide_df = wide_df.append(tmp_dt, ignore_index=True)
        tmp_df = tmp_df.shift(1)

    return wide_df

def ts_windowing_group(df, window_size, ts_columns, grp_columns):
    tmp_df = df
    tot_ts_columns_seq = []

    for idx in range(window_size, 0, -1):
        idx_suffix = 't-' + str(idx)
        ts_columns_seq = [x + '_' + idx_suffix for x in ts_columns]
        tot_ts_columns_seq.extend(ts_columns_seq)
        tmp_df[ts_columns_seq] = tmp_df.groupby(grp_columns)[ts_columns].shift(idx)

    tmp_df.dropna(subset=tot_ts_columns_seq, inplace=True)

    return tmp_df, tot_ts_columns_seq

def umap_proj(neighbor=10, n_comp=2, min_dist=0.1, X=None):
    @numba.njit()
    def sqeuclidean(a,b):
        a_b = a - b
        return np.dot(a_b, a_b)

    return umap.UMAP(n_neighbors=neighbor,
                     n_components=n_comp,
                     min_dist=min_dist,
                     metric=sqeuclidean).fit_transform(X)