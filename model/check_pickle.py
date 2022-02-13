import pickle as pkl
train_pkl = "train_tmp_data.pkl"
val_pkl = "val_tmp_data.pkl"

with open(train_pkl, "rb") as file:
    train_data = pkl.load(file)
with open(val_pkl, "rb") as file:
    val_data = pkl.load(file)

print(f'length of train data: {len(train_data)}')
print(f'length of validation data:{len(val_data)}')