# This pythons script is used to convert tensorboard data to csv file

# access to tensorboard https://tensorboard.dev/experiment/JfMZJUwdReC8FPp83s3RDg/

import pandas as pd
import os
import tensorboard as tb
import pathlib

experiment_id = (  # multi 2 bins'Goh3jdHwQpedTjWqAxsgoQ' #multi affinity  # depth+ pos_enc"5Jh1z944SBCGobZLLmQdwg" # depth: "K1NKDNUYTqK9GaGVtWvnCA";'w5mbDYCmS5Oz0gYqbw1qLA' # angular loss : 'n05xObObQU24MjsM7KnRRQ'； "w5mbDYCmS5Oz0gYqbw1qLA" 'zgVmzRRNSa2q2smYZClHjQ';multi four bins iJvlXxzgRFKSZwQT5JefxA;exp-A 1mnIdo5MToy3bLZxfDunDw/
    '1mnIdo5MToy3bLZxfDunDw'
)
csv_file_dir = 'csv_output'
csv_file_prefix = 'exp-A'


def clean_data(input_df, csv_name):
    # remove accuracy tag and wall time columns
    input_df = input_df.drop(columns=['tag', 'wall_time'])
    # remove data in "run" columns
    input_df['run'] = input_df['run'].str.extract(r'(\D*)_\d\d\d\d')
    cleaned_df = pd.pivot_table(input_df, values='value', index='step', columns='run')
    # export to csv
    output_dir = pathlib.Path(csv_file_dir)
    cleaned_df.to_csv(output_dir / f'{csv_file_prefix}_{csv_name}', index=True)
    print("Finish Exporting ", csv_name)


if __name__ == '__main__':
    # retrieve tensorboard data by experiment_id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars(include_wall_time=True)
    print(df["tag"].unique())
    # split into loss and accuracy datafram
    df_acc = df[df['tag'] == 'epoch_orientation_accuracy']
    df_loss = df[df['tag'] == 'epoch_loss']
    # further split into validation accuracy, training accuracy, validation loss and training loss
    '''
    val_acc = df_acc[df.run.str.endswith("\\validation")]
    train_acc = df_acc[df.run.str.endswith("\\train")]
    val_loss = df_loss[df.run.str.endswith("\\validation")]
    train_loss = df_loss[df.run.str.endswith("\\train")]
    '''
    val_acc = df_acc[df['run'].str.endswith("validation")]
    train_acc = df_acc[df['run'].str.endswith("train")]
    val_loss = df_loss[df['run'].str.endswith("validation")]
    train_loss = df_loss[df['run'].str.endswith("train")]

    process_queue = [
        [val_acc, "cleaned_validation_accuracy.csv"],
        [train_acc, 'cleaned_training_accuracy.csv'],
        [val_loss, "cleaned_validation_loss.csv"],
        [train_loss, "cleaned_training_loss.csv"],
    ]
    for (dataframe, export_file_name) in process_queue:
        clean_data(dataframe, export_file_name)
    print("Finish Processing Queue")
