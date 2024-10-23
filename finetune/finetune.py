import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

# Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 1. 自定义数据集
class Food2kDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        # 读取图像路径
        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.images = [line.strip() for line in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.root_dir + self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 从图像路径中提取标签ID
        label_id = int(self.images[idx].split('/')[-2])

        if self.transform:
            image = self.transform(image)

        return image, label_id

# 2. 加载预训练的DINOv2模型
def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    return model

# 3. 修改模型以适应新任务
class FineTunedDINOv2(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.num_features, num_classes)
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

# 4. 准备数据
def get_data(root_dir, train_file, test_file, batch_size=32, rank=0, world_size=1):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = Food2kDataset(root_dir, train_file, transform=transform)
    test_dataset = Food2kDataset(root_dir, test_file, transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=64)
    
    return train_loader, test_loader

# 5. 训练函数
def train(rank, world_size, args, config):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 加载基础模型
    base_model = load_model()
    
    # 冻结基础模型的参数
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 创建微调模型
    num_classes = args.num_classes
    model = FineTunedDINOv2(base_model, num_classes).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 准备数据
    train_loader, test_loader = get_data(args.root_dir, args.train_file, args.test_file, args.batch_size, rank, world_size)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.module.classifier.parameters(),
        lr=config['optim']['base_lr'],
        betas=(config['optim']['adamw_beta1'], config['optim']['adamw_beta2']),
        weight_decay=config['optim']['weight_decay']
    )
    
    best_accuracy = 0.0
    best_model_path = None

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(config['optim']['epochs']):
        model.train()
        running_loss = 0.0
        train_loader.sampler.set_epoch(epoch)
        
        # Only create a progress bar for rank 0
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch+1}/{config['optim']['epochs']}", leave=False)
        else:
            progress_bar = train_loader  # Use the loader directly for other ranks

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar only for rank 0
            if rank == 0:
                progress_bar.set_postfix(loss=running_loss/len(train_loader))
        
        if rank == 0:
            print(f"Rank {rank} Epoch {epoch+1}/{config['optim']['epochs']}, Loss: {running_loss/len(train_loader)}")
        
        # Save the model for this epoch
        if rank == 0:  # Only save from the main process
            current_model_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), current_model_path)
            
            # Evaluate the model
            accuracy = evaluate(model, test_loader, device)
            
            # Check if this is the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = current_model_path

    # Rename the best model
    if rank == 0 and best_model_path:
        best_model_final_path = os.path.join(args.save_dir, 'best_model.pth')
        os.rename(best_model_path, best_model_final_path)
        print(f"Best model saved as '{best_model_final_path}' with accuracy: {best_accuracy:.2f}%")

    dist.destroy_process_group()

# 6. 评估函数
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on Food2k dataset')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--train_file', type=str, required=True, help='Training file name')
    parser.add_argument('--test_file', type=str, required=True, help='Testing file name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    world_size = args.num_gpus
    local_rank = int(os.environ['LOCAL_RANK'])  # Read the local rank from the environment variable
    train(local_rank, world_size, args, config)

if __name__ == '__main__':
    main()