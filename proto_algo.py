import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import random
from collections import defaultdict
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Few-Shot Dataset with Image Filtering
class FewShotImageFolder(Dataset):
    def __init__(self, root, transform=None, augment_underrepresented=False):
        self.root = root
        self.transform = transform
        self.class_to_imgs = defaultdict(list)
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
        # Gather images per class, filter non-images
        for cls in self.classes:
            cls_folder = os.path.join(root, cls)
            for img in os.listdir(cls_folder):
                if os.path.splitext(img)[1].lower() in IMG_EXTS:
                    self.class_to_imgs[cls].append(os.path.join(cls_folder, img))
        # Data augmentation for underrepresented classes
        if augment_underrepresented:
            class_sizes = [len(self.class_to_imgs[cls]) for cls in self.classes]
            max_size = max(class_sizes)
            for cls in self.classes:
                imgs = self.class_to_imgs[cls]
                if len(imgs) < max_size and len(imgs) > 0:
                    extra_imgs = random.choices(imgs, k=(max_size - len(imgs)))
                    self.class_to_imgs[cls].extend(extra_imgs)

    def __len__(self):
        return sum([len(v) for v in self.class_to_imgs.values()])

    def __getitem__(self, idx):
        # Not used directly, use get_episode
        raise NotImplementedError

    def get_episode(self, n_way, k_shot, q_query):
        selected_classes = random.sample(self.classes, n_way)
        support_x, support_y, query_x, query_y = [], [], [], []
        for i, cls in enumerate(selected_classes):
            imgs = self.class_to_imgs[cls]
            chosen = random.sample(imgs, k_shot + q_query)
            support = chosen[:k_shot]
            query = chosen[k_shot:]
            for s in support:
                im = Image.open(s).convert('RGB')
                if self.transform:
                    im = self.transform(im)
                support_x.append(im)
                support_y.append(i)
            for q in query:
                im = Image.open(q).convert('RGB')
                if self.transform:
                    im = self.transform(im)
                query_x.append(im)
                query_y.append(i)
        return torch.stack(support_x), torch.tensor(support_y), torch.stack(query_x), torch.tensor(query_y)

# 2. Lightweight CNN Encoder
class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(x_dim, hid_dim, 3, padding=1), nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hid_dim, hid_dim, 3, padding=1), nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hid_dim, z_dim, 3, padding=1), nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
    def forward(self, x):
        return self.encoder(x)

def get_output_size(model, input_shape):
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        out = model(x)
    return out.shape[-1]

# 3. Prototypical Network
class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query):
        n_way = torch.unique(support_labels).size(0)
        z_support = self.encoder(support)
        z_query = self.encoder(query)
        prototypes = []
        for c in range(n_way):
            prototypes.append(z_support[support_labels == c].mean(0))
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(z_query, prototypes)
        scores = -dists
        return scores

# 4. Training & Evaluation Loop
def train_protonet(
        dataset, n_way, k_shot, q_query, epochs, episodes_per_epoch, device, val_dataset=None, eval_freq=1
    ):
    encoder = ConvNet()
    model = ProtoNet(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    acc_hist = []

    for epoch in range(epochs):
        for episode in range(episodes_per_epoch):
            support_x, support_y, query_x, query_y = dataset.get_episode(n_way, k_shot, q_query)
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)
            scores = model(support_x, support_y, query_x)
            loss = loss_fn(scores, query_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate at end of each epoch
        if val_dataset is not None and (epoch+1) % eval_freq == 0:
            val_acc = evaluate_protonet(model, val_dataset, n_way, k_shot, q_query, device, n_episodes=10)
            acc_hist.append(val_acc)
            print(f"Epoch {epoch+1}/{epochs}, Val Accuracy: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} done.")
    return model, acc_hist

def evaluate_protonet(model, dataset, n_way, k_shot, q_query, device, n_episodes=10):
    model.eval()
    accs, preds, trues, embs, labels = [], [], [], [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            support_x, support_y, query_x, query_y = dataset.get_episode(n_way, k_shot, q_query)
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)
            scores = model(support_x, support_y, query_x)
            pred = scores.argmax(1).cpu().numpy()
            true = query_y.cpu().numpy()
            acc = (pred == true).mean()
            accs.append(acc)
            preds.extend(pred)
            trues.extend(true)
            embs.append(model.encoder(query_x).cpu().numpy())
            labels.extend(true)
    acc = np.mean(accs)
    precision = precision_score(trues, preds, average='macro', zero_division=0)
    recall = recall_score(trues, preds, average='macro', zero_division=0)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    cm = confusion_matrix(trues, preds)
    print(f"Acc: {acc:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    # t-SNE visualization
    try:
        embs = np.concatenate(embs, axis=0)
        tsne = TSNE(n_components=2)
        embs_2d = tsne.fit_transform(embs)
        plt.figure(figsize=(8,6))
        for i in range(n_way):
            inds = np.array(labels) == i
            plt.scatter(embs_2d[inds,0], embs_2d[inds,1], label=f"Class {i}")
        plt.legend()
        plt.title("t-SNE of Query Embeddings")
        plt.show()
    except Exception as e:
        print(f"t-SNE plot failed: {e}")
    model.train()
    return acc

# 5. Main Execution
if __name__ == "__main__":
    DATASET_PATH = "D:/Shiga/3d defects"  # Folder: class_name/*.jpg
    N_WAY = 6
    K_SHOT = 5
    Q_QUERY = 10
    EPOCHS = 5
    EPISODES_PER_EPOCH = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    transform = T.Compose([
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = FewShotImageFolder(DATASET_PATH, transform=transform, augment_underrepresented=True)
    # Optionally split into train/val/test as needed
    # For demo, use same for train and val
    model, acc_hist = train_protonet(dataset, N_WAY, K_SHOT, Q_QUERY, EPOCHS, EPISODES_PER_EPOCH, DEVICE, val_dataset=dataset)

    print("Final Evaluation:")
    evaluate_protonet(model, dataset, N_WAY, K_SHOT, Q_QUERY, DEVICE, n_episodes=20)

    # Binary classification extension (defective vs. non-defective)
    # Assuming 'Normal' is class 0, others are defective
    def to_binary(y, normal_class_index=0):
        return (y != normal_class_index).astype(np.int64)

    # For evaluation:
    # binary_trues = to_binary(np.array(trues))
    # binary_preds = to_binary(np.array(preds))
    # print("Binary Acc:", accuracy_score(binary_trues, binary_preds), ...)