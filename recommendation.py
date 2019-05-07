#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType


spark = SparkSession.builder.appName('big_d_recommendation_bois').getOrCreate()

df_all = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
df = df_all.sample(False, .01, 12345)

users = df_all.select('user_id')
users = users.repartition(2000)
tracks = df_all.select('track_id')
tracks = tracks.repartition(2000)

user_indexer = StringIndexer(inputCol="user_id", outputCol="userid_int")
fit_user_indexer = user_indexer.fit(users)

track_indexer = StringIndexer(inputCol="track_id", outputCol="trackid_int")
fit_track_indexer = track_indexer.fit(tracks)
    
fit_user_indexer.save('user_indexer')
fit_track_indexer.save('track_indexer')

df_all_out = fit_track_indexer.transform(df_all)
df_all_out = fit_user_indexer.transform(df_all_out)

df_all_out.repartition(2000).write.parquet("transformed_cf_train.parquet")


df_x = fit_track_indexer.transform(df)
df_final = fit_user_indexer.transform(df_x)

# cuz string indexer converts to double
df_final = df_final.withColumn("userid_int", df_final["userid_int"].cast(IntegerType()))
df_final = df_final.withColumn("trackid_int", df_final["trackid_int"].cast(IntegerType()))

df_final.createOrReplaceTempView("tracks_data")

df = spark.sql("Select userid_int as user, trackid_int as item, count as rating FROM tracks_data")

print('training ALS model')

als = ALS(rank=10,
          maxIter=10,
          regParam=0.01,
          implicitPrefs=True,
          userCol="user",
          itemCol="item",
          ratingCol="rating")
als.fit(df)
als.save("als_model")
