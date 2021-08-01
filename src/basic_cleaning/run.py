#!/usr/bin/env python
"""
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import os
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info('Artifact downloaded successfully')
    
    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded Dataframe")

    indices = df['price'].between(args.min_price, args.max_price)
    df = df[indices].copy()
    indices = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)  
    df = df[indices].copy()
    logger.info("Outliers removed")

    # last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Converted last_review to datetime format")
    
    df.to_csv(args.output_artifact, index=False)
    logger.info('Saved Dataframe successfully')

    # Create an artifact instance
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    os.remove(args.output_artifact)
    logger.info('Artifact successfully uploaded to W&B!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='input artifact name',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='output artifact name',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='output artifact type',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='Description of artifact',
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min rental price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Max rental price",
        required=True
    )


    args = parser.parse_args()

    go(args)
