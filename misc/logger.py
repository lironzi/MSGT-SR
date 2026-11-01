import torch
import os

class Logger():
    def __init__(self,path):
        self.path=path
        self.file_name="train_log.txt"


    def save_checkpoint(self,model, optimizer, epoch, loss, path="checkpoint.pth"):

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, path)

    def log_epoch(self,epoch, train_loss, val_loss):
        line = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"
        print(line.strip())
        with open(self.path / self.file_name, "a") as f:
            f.write(line)

    def log_string(self,str):
        print(str)
        with open(self.path / self.file_name, "a") as f:
            f.write(str)

