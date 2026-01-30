from utils.dataset import PatchDataset

dataset = PatchDataset(
    "data/archive/TRAINING_CG-1050\TRAINING/ORIGINAL",
    label=0
)

print("Total patches:", len(dataset))
patch, label = dataset[0]

print("Patch shape:", patch.shape)
print("Label:", label)
