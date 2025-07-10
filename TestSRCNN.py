from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import glob, os, re
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ----------- Preprocess One Image for Visualization -----------
def preprocess_image(path, output_size=(512, 512)):
    img = Image.open(path).convert('RGB')
    img = img.resize(output_size, resample=Image.BICUBIC)
    downscale_size = (output_size[0] // 2, output_size[1] // 2)
    downscale = transforms.Resize(downscale_size, interpolation=Image.BICUBIC)
    low_res = downscale(img)
    to_tensor = transforms.ToTensor()
    return to_tensor(low_res), to_tensor(img)

# ----------- Dataset Class -----------
class DIV2KDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.png'))
        self.hr_size = (512, 512)
        self.lr_size = (self.hr_size[0] // 2, self.hr_size[1] // 2)

        self.hr_transform = transforms.Compose([
            transforms.Resize(self.hr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

# ----------- Model -----------
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ----------- Auto-load Latest Checkpoint -----------
def get_latest_checkpoint(folder="SRCNN", pattern="srcnn_checkpoint_epoch(\d+)\.pth"):
    files = glob.glob(os.path.join(folder, "srcnn_checkpoint_epoch*.pth"))
    latest_epoch = -1
    latest_file = None
    for f in files:
        match = re.search(pattern, os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f
    return latest_file

# ----------- Calculate SSIM and PSNR -----------
def calculate_metrics(pred, gt):
    pred_np = pred.permute(1, 2, 0).numpy()
    gt_np = gt.permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    ssim_val = ssim(gt_np, pred_np, channel_axis=2, data_range=1)
    psnr_val = psnr(gt_np, pred_np, data_range=1)
    return ssim_val, psnr_val

# ============== MAIN RUN BLOCK ==============
if __name__ == "__main__":
    image_path = glob.glob(r'C:\Users\Dell\Documents\image_sharpening_project\data\input_images\*.png')[0]
    low_res_tensor, high_res_tensor = preprocess_image(image_path)

    dataset_path = r'C:\Users\Dell\Documents\image_sharpening_project\data\input_images'
    train_dataset = DIV2KDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, prefetch_factor=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    latest_ckpt = get_latest_checkpoint("SRCNN")
    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt))
        start_epoch = int(re.search(r'epoch(\d+)', latest_ckpt).group(1))
        print(f"Loaded checkpoint: {latest_ckpt} (resuming from epoch {start_epoch})")
    elif os.path.exists("SRCNN/srcnn_checkpoint.pth"):
        model.load_state_dict(torch.load("SRCNN/srcnn_checkpoint.pth"))
        print("Loaded fallback checkpoint: srcnn_checkpoint.pth")
    else:
        print("No previous checkpoint found. Starting fresh training.")

    num_epochs = 10

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for low_res_imgs, high_res_imgs in train_loader:
            batch_count += 1
            low_res_imgs = low_res_imgs.to(device)
            high_res_imgs = high_res_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(low_res_imgs)
            loss = criterion(outputs, high_res_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.8f}")

        avg_loss = running_loss / batch_count
        print(f"Epoch [{epoch+1}] completed. Avg Loss: {avg_loss:.8f}")
        torch.cuda.empty_cache()

        # Save checkpoint
        os.makedirs("SRCNN", exist_ok=True)
        ckpt_filename = os.path.join("SRCNN", f"srcnn_checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_filename)
        torch.save(model.state_dict(), os.path.join("SRCNN", "srcnn_checkpoint.pth"))
        print(f"Saved checkpoint: {ckpt_filename}")

        # Save visual output + metrics
        model.eval()
        os.makedirs(f"output_images/epoch_{epoch+1}", exist_ok=True)

        with torch.no_grad():
            for i, (lr, hr) in enumerate(train_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model(lr)

                for j in range(min(lr.shape[0], 4)):
                    lr_img = lr[j].cpu().clamp(0, 1)
                    sr_img = sr[j].cpu().clamp(0, 1)
                    hr_img = hr[j].cpu().clamp(0, 1)

                    lr_img_up = torch.nn.functional.interpolate(
                        lr_img.unsqueeze(0),
                        scale_factor=2,
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)

                    ssim_val, psnr_val = calculate_metrics(sr_img, hr_img)
                    print(f"Epoch {epoch+1}, Sample {j+1} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

                    composite = torch.cat([lr_img_up, sr_img, hr_img], dim=2)
                    composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    composite_pil = Image.fromarray(composite_np)
                    composite_pil.save(f"output_images/epoch_{epoch+1}/sample_{j+1}.png")

                break  # only one batch for evaluation
