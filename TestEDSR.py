import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob, os, re
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ----------- Helper: Center crop to square -----------
def center_crop_to_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))

# ----------- Dataset Class -----------
class DIV2KDataset(Dataset):
    def __init__(self, folder_path, img_size=512):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.png'))
        print(f"Found {len(self.image_paths)} images in dataset folder.")
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = center_crop_to_square(img)
        img = self.resize(img)
        img_tensor = self.to_tensor(img)
        return img_tensor, img_tensor

# ----------- EDSR Model -----------
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, num_blocks=8, num_features=64, scale=2):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

# ----------- Calculate SSIM and PSNR -----------
def calculate_metrics(pred, gt):
    pred_np = pred.permute(1, 2, 0).numpy()
    gt_np = gt.permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    ssim_val = ssim(gt_np, pred_np, channel_axis=2, data_range=1)
    psnr_val = psnr(gt_np, pred_np, data_range=1)
    return ssim_val, psnr_val

# ----------- Modified Auto-load Checkpoint ----------- #
def get_latest_checkpoint():
    epoch_ckpts = glob.glob(r'C:\Users\Dell\Documents\image_sharpening_project\EDSR\edsr_checkpoint_epoch*.pth')
    if epoch_ckpts:
        latest_epoch = -1
        latest_file = None
        for f in epoch_ckpts:
            match = re.search(r'epoch(\d+)', os.path.basename(f))
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = f
        return latest_file, latest_epoch
    elif os.path.exists("edsr_checkpoint.pth"):
        return "edsr_checkpoint.pth", 0
    else:
        return None, 0

# ============== MAIN RUN BLOCK ============== #
if __name__ == "__main__":
    dataset_path = r"C:\Users\Dell\Documents\image_sharpening_project\data\input_images"
    output_base_path = r"C:\Users\Dell\Documents\image_sharpening_project\output_images\EDSR"

    train_dataset = DIV2KDataset(dataset_path, img_size=512)
    print(f"Dataset length: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EDSR(num_blocks=4, num_features=32, scale=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    latest_ckpt, start_epoch = get_latest_checkpoint()
    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt))
        print(f"Loaded checkpoint: {latest_ckpt} (resuming from epoch {start_epoch})")
    else:
        print("No previous checkpoint found. Starting fresh training.")

    num_epochs = 5
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        batch_found = False

        for batch_count, (low_res_imgs, high_res_imgs) in enumerate(train_loader, 1):
            batch_found = True
            print(f"Processing batch {batch_count}")
            low_res_imgs = low_res_imgs.to(device)
            high_res_imgs = high_res_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(low_res_imgs)
            high_res_upscaled = nn.functional.interpolate(high_res_imgs, scale_factor=2, mode='bicubic', align_corners=False)
            loss = criterion(outputs, high_res_upscaled)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.8f}")

        if not batch_found:
            print("No batches found in DataLoader! Check your dataset folder and images.")
            break

        avg_loss = running_loss / batch_count
        print(f"Epoch [{epoch+1}] completed. Avg Loss: {avg_loss:.8f}")
        torch.cuda.empty_cache()

        # -------- Save checkpoints with epoch -------- #
        checkpoint_dir = r"C:\Users\Dell\Documents\image_sharpening_project\EDSR"
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure folder exists

        epoch_ckpt_path = os.path.join(checkpoint_dir, f"edsr_checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)
        latest_ckpt_path = os.path.join(checkpoint_dir, "edsr_checkpoint.pth")  # Always overwrite latest
        torch.save(model.state_dict(), latest_ckpt_path)
        print(f"Saved checkpoint: {epoch_ckpt_path}")

        # -------- Save sample outputs -------- #
        model.eval()
        save_path = os.path.join(output_base_path, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for i, (lr, hr) in enumerate(train_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model(lr)
                for j in range(min(lr.shape[0], 4)):
                    lr_img = lr[j].cpu().clamp(0, 1)
                    sr_img = sr[j].cpu().clamp(0, 1)
                    hr_img = hr[j].cpu().clamp(0, 1)

                    lr_img_up = nn.functional.interpolate(lr_img.unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False).squeeze(0)
                    hr_img_up = nn.functional.interpolate(hr_img.unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False).squeeze(0)

                    ssim_val, psnr_val = calculate_metrics(sr_img, hr_img_up)
                    print(f"Epoch {epoch+1}, Sample {j+1} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

                    composite = torch.cat([lr_img_up, sr_img, hr_img_up], dim=2)
                    composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(composite_np).save(os.path.join(save_path, f"sample_{j+1}.png"))
                break