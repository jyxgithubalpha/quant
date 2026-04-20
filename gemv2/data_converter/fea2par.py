import os
import polars as pl
import pyarrow.parquet as pq
from datetime import datetime


def standardize_date_and_code(df):
    df = df.with_columns(
        pl.col("date").cast(pl.String)
    )
    df = df.with_columns(
        pl.col("Code").cast(pl.String)
    )
    df = df.sort("date", "Code")
    return df

def fac_fea2par_converter(src_path, tar_dir):
    df = pl.read_ipc(src_path, memory_map=False)
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")

def label_fea2par_converter(src_path, tar_dir):
    df = pl.read_ipc(src_path, memory_map=False)
    df = df.rename({"index": "date"}).unpivot(index="date", variable_name="Code", value_name="label")
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")

def liquidity_fea2par_converter(src_path, tar_dir):
    df = pl.read_ipc(src_path, memory_map=False)
    df = df.rename({"index": "date"}).unpivot(index="date", variable_name="Code", value_name="liquidity", )
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")

def bench_fea2par_converter(src_path, tar_dir, bench_name):
    df = pl.read_ipc(src_path, memory_map=False)
    df = df.unpivot(index="date", variable_name="Code", value_name=bench_name)
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")

def benchmark_fea2par_converter(src_path, tar_dir, bench_name):
    df = pl.read_ipc(src_path, memory_map=False)
    df = df.unpivot(index="date", variable_name="Code", value_name=bench_name)
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")

def score_fea2par_converter(src_path, tar_dir, score_name):
    df = pl.read_ipc(src_path, memory_map=False)
    df = df.unpivot(index="date", variable_name="Code", value_name=score_name)
    df = standardize_date_and_code(df)
    table = df.to_arrow()
    pq.write_to_dataset(table, tar_dir, partition_cols=["date"], compression="snappy")


def main():
    par_dir = rf"/home/user165/workspace/data/parquet/"
    
    fac_src_path = rf"/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
    fac_tar_dir = os.path.join(par_dir, "fac")
    # fac_fea2par_converter(fac_src_path, fac_tar_dir)

    label_src_path = rf"/project/model_share/share_1/label_data/label1.fea"
    label_tar_dir = os.path.join(par_dir, "label")
    label_fea2par_converter(label_src_path, label_tar_dir)

    liquidity_src_path = rf"/project/model_share/share_1/label_data/can_trade_amt1.fea"
    liquidity_tar_dir = os.path.join(par_dir, "liquidity")
    liquidity_fea2par_converter(liquidity_src_path, liquidity_tar_dir)

    bench_configs = {
        "bench1": "/project/model_share_remote/score_file/sub_score/label1/bench1_20250822.fea",
        "bench2": "/project/model_share_remote/score_file/sub_score/label1/bench2_20250822.fea", 
        "bench3": "/project/model_share_remote/score_file/sub_score/label1/bench3_20250822.fea",
        "bench4": "/project/model_share_remote/score_file/sub_score/label1/bench4_20250822.fea",
        "bench5": "/project/model_share_remote/score_file/bench5_label1.fea",
        "bench6": "/home/user165/workspace/data/label1_bench6.fea",
    }
    
    for bench_name, src_path in bench_configs.items():
        tar_dir = os.path.join(par_dir, bench_name)
        bench_fea2par_converter(src_path, tar_dir, bench_name)

    
    benchmark_src_path = rf"/project/model_share_remote/score_file/benchmark_1_0819.fea"
    benchmark_tar_dir = os.path.join(par_dir, "benchmark")
    benchmark_fea2par_converter(benchmark_src_path, benchmark_tar_dir, "benchmark")


    score_src_path = rf"/project/model_share_remote/score_file/label_1_benchmark_20250619.fea"
    score_tar_dir = os.path.join(par_dir, "score")
    score_fea2par_converter(score_src_path, score_tar_dir, "score")


if __name__ == "__main__":
    main()