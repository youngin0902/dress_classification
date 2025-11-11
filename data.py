from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# === 증강 정의 ===
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),              # 1️⃣ 큰 이미지를 512x512로 축소
    transforms.RandomHorizontalFlip(p=0.5),     # 2️⃣ 좌우 반전
    transforms.RandomRotation(10),              # 3️⃣ ±10도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 4️⃣ 밝기/대비 변형
    transforms.ToTensor(),                      # 5️⃣ 텐서 변환 (0~1)
    transforms.Normalize(                       # 6️⃣ 정규화 (대략적 값 사용)
        mean=[0.66, 0.66, 0.66],
        std=[0.13, 0.13, 0.13]
    )
])

# === 검증용 변환 (증강 X, 크기만 맞춤) ===
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.66, 0.66, 0.66], std=[0.13, 0.13, 0.13])
])

# === ImageFolder 로드 ===
train_data = datasets.ImageFolder(root="data/dress/train", transform=train_transform)
test_data = datasets.ImageFolder(root="data/dress/test", transform=val_transform)

# === DataLoader ===
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
