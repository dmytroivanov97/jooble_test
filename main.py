import re

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import mean, stddev, udf, col, array
from pyspark.sql.types import IntegerType

app_name = 'jooble_test'
path = ""
result_filename = 'test_transformed.csv'


def standardize_func(mean_val, stddev_val) -> object:
    def func(col_val) -> float:
        return (col_val - mean_val) / stddev_val

    return func


def read_dataframes(spark: SparkSession):
    train_df = spark.read.csv(path + 'train.csv', header=True, inferSchema=True)
    test_df = spark.read.csv(path + 'test.csv', header=True, inferSchema=True)
    return train_df, test_df


def get_mean_df(df: DataFrame, col_feature_list: list):
    return df.agg(*[mean(c).alias(c) for c in df.columns if c in col_feature_list])


def get_stddevs_df(df: DataFrame, col_feature_list: list):
    return df.agg(*[stddev(c).alias(c) for c in df.columns if c in col_feature_list])


def get_standardization_df(factor_number: int, test_df: DataFrame, means_df: DataFrame, stddevs_df: DataFrame,
                           count_col: int):
    for i in range(count_col):
        mean_val = means_df.first()[f"feature_type_{factor_number}_{i}"]
        stddev_val = stddevs_df.first()[f"feature_type_{factor_number}_{i}"]
        stand_func = standardize_func(mean_val, stddev_val)
        test_df = test_df.withColumn(f"feature_type_{factor_number}_stand_{i}",
                                     stand_func(test_df[f"feature_type_{factor_number}_{i}"])).drop(
            f"feature_type_{factor_number}_{i}")
    return test_df


def add_max_feature_type_index(factor_number: int, df: DataFrame, count_col: int):
    max_index_udf = udf(lambda x: x.index(max(x)), IntegerType())
    return df.withColumn(f"max_feature_type_{factor_number}_index", max_index_udf(
        array([col(f"feature_type_{factor_number}_stand_{i}") for i in range(count_col)])))


def main():
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    train_df, test_df = read_dataframes(spark=spark)
    feature_cols_all = [c for c in train_df.columns if c.startswith('feature_type')]
    count_of_factors = set(map(lambda x: int(re.findall(r'feature_type_(\d+)_', x)[0]), feature_cols_all))

    for factor_number in count_of_factors:
        pattern = f'feature_type_{factor_number}_(\d+)'
        feature_cols = [c for c in feature_cols_all if re.match(pattern, c)]
        count_columns = len(feature_cols)

        means = get_mean_df(train_df, feature_cols)
        stddevs = get_stddevs_df(train_df, feature_cols)

        test_df = get_standardization_df(factor_number, test_df, means, stddevs, count_columns)
        test_df = add_max_feature_type_index(factor_number, test_df, count_columns)

    test_df.toPandas().to_csv(path + result_filename, header=True)


if __name__ == "__main__":
    main()
