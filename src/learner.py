import os 
import torch
import pandas as pd
from typing import Any, Dict, List, Tuple

from torch.nn import Module
from torch.utils.data import DataLoader
from src.dataloader import data_collector


class Learner():
    def __init__(self, 
        args: Any, 
        model: Module, 
        metric: Any,
        device: Any, 
        save_path=None,
    ):
        self.args = args
        self.model = model
        self.metric = metric
        self.device = device
        self.best_score = -100
        self.early_stopping_counter = 0
        self.save_path = save_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def save_checkpoint(self, f1_score: float):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "f1_score": f1_score
        }
        torch.save(checkpoint, f"{self.args.checkpoint_dir}/model.pt")

    def load_checkpoint(self, checkpoint_path: str):
        if checkpoint_path is None:
            return None
        if not os.path.exists(checkpoint_path):
            return None

        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_score = checkpoint["f1_score"]

    def evaluate(self, dataset, save_path=None, device='cuda'):
        id2l = dataset.id2l
        scheme = dataset.scheme

        dataloader = DataLoader( 
            dataset, batch_size=32, 
            shuffle=False, collate_fn=data_collector
        )
        results = []
        accumulate_pred = []
        accumulate_true = []
        total_gpu_ms = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_features = ['input_ids', 'attention_mask']
                features = {key:val.to(device) for key,val in batch.items() if key in input_features}

                start, end = self._gpu_events()
                if start:
                    start.record()
                (logits, _) = self.model(features)
                if end:
                    end.record()
                total_gpu_ms += self._elapsed_ms(start, end)

                for ids in range(len(batch['input_ids'])):
                    mask = batch['attention_mask'][ids]
                    filter = len(mask[mask == 1])
                    labels = batch['type_ids'][ids][:filter].cpu().tolist()
                    logit = logits[ids][:filter].argmax(-1).cpu().tolist()
                    words = batch['input_ids'][ids][:filter].cpu().tolist()
                    preds = [id2l[x] for x in logit]
                    labels = [id2l[x] for x in labels]

                    temp = []
                    for idw in range(len(words)):
                        word = dataset.text_processor.tok.decode([words[idw]])
                        tmp_pred = "O" if preds[idw] == "O" else f"{preds[idw]}"
                        tmp_label = "O" if labels[idw] == "O" else f"{labels[idw]}"
                        accumulate_pred.append(tmp_pred)
                        accumulate_true.append(tmp_label)
                        temp.append([word, tmp_label, tmp_pred])
                    results.append(temp)

        n_samples = sum(len(b) for b in results)
        print(f"Inference GPU time: {total_gpu_ms/1000:.2f}s  ({total_gpu_ms/max(len(dataloader),1):.1f}ms/batch, {total_gpu_ms/max(n_samples,1):.2f}ms/sample)")

        # Report type metrics
        report = self.metric.compute(
            references=[accumulate_true], 
            predictions=[accumulate_pred], 
            scheme=scheme
        )

        score = pd.DataFrame({
            'index': ['TYPE'], 'precision': ['-'], 'recall': ['-'],
            'f1': [report['overall_f1']], 'number': ['-'], 'accuracy': ['-'],
            'gpu_time_s': [round(total_gpu_ms / 1000, 4)],
            'gpu_ms_per_batch': [round(total_gpu_ms / max(len(dataloader), 1), 4)],
            'gpu_ms_per_sample': [round(total_gpu_ms / max(n_samples, 1), 4)],
        })

        # Save the results
        if save_path is not None:
            filename = dataset.filename.split('.')[0]
            score.to_csv(save_path + f"/total_report_{filename}.csv", index=False)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path + f"/results_{filename}.txt", "w") as f:
                for item in results:
                    for word, label, pred in item:
                        f.write(f"{word} {label} {pred}\n")
                    f.write("\n")

            with open(save_path + f"/report_{filename}.txt", "w") as f:
                for key, value in report.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                f.write(f"gpu_time_s: {total_gpu_ms/1000:.4f}\n")
                f.write(f"gpu_ms_per_batch: {total_gpu_ms/max(len(dataloader),1):.4f}\n")
                f.write(f"gpu_ms_per_sample: {total_gpu_ms/max(n_samples,1):.4f}\n")

        return report['overall_f1'], (score, report, results)
    

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(logits.device)
        mask = mask.flatten(start_dim=0, end_dim=1)
        labels = labels.flatten().to(logits.device)
        logits = logits.flatten(start_dim=0, end_dim=1)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = criterion(logits, labels)*mask
        return loss.sum() / mask.sum().float()

    def predict(self, dataset) -> List[Dict[str, Any]]:
        return []

    def _gpu_events(self):
        if self.device.type == 'cuda':
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        return None, None

    def _elapsed_ms(self, start, end):
        if start is None:
            return 0.0
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def train_step(self, batch: Tuple[torch.Tensor]) -> Tuple[float, float]:
        self.model.train()
        input_features = ['input_ids', 'attention_mask']
        features = {k: batch[k].to(self.device) for k in input_features}

        start, end = self._gpu_events()
        if start:
            start.record()

        logits, _ = self.model(features)
        loss = self.compute_loss(logits, batch['type_ids'], batch['attention_mask'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if end:
            end.record()

        return loss.item(), self._elapsed_ms(start, end)

    def train(self, train_dataset, dev_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=data_collector
        )
        self.early_stopping_counter = 0
        total_steps = len(train_loader) * self.args.epochs
        steps_per_epoch = total_steps // self.args.epochs
        eval_step_size = steps_per_epoch // self.args.eval_per_epoch
        epoch_records = []

        for epoch in range(self.args.epochs):
            total_loss = 0
            total_gpu_ms = 0.0
            for index, batch in enumerate(train_loader):
                loss, gpu_ms = self.train_step(batch)
                total_gpu_ms += gpu_ms

                if index % 500 == 0:
                    print(f"Epoch: {epoch}, Step: {str(index):7s}, Loss: {loss}")

                total_loss += loss
                if (index + 1) % eval_step_size == 0:
                    f1_score, _ = self.evaluate(dev_dataset)
                    if f1_score > self.best_score:
                        self.best_score = f1_score
                        self.save_checkpoint(f1_score)
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter > self.args.early_stopping*self.args.eval_per_epoch:
                            self.load_checkpoint(self.args.checkpoint_dir)
                            self._save_training_time_report(epoch_records)
                            return

            avg_loss = total_loss / len(train_loader)
            n_steps = len(train_loader)
            epoch_records.append({
                'epoch': epoch,
                'avg_loss': round(avg_loss, 6),
                'gpu_time_s': round(total_gpu_ms / 1000, 4),
                'gpu_ms_per_step': round(total_gpu_ms / max(n_steps, 1), 4),
            })
            print(f"Epoch: {epoch}, Average Loss: {avg_loss}, GPU Train Time: {total_gpu_ms/1000:.2f}s")
            self.scheduler.step(avg_loss)

        self.load_checkpoint(self.args.checkpoint_dir)
        self._save_training_time_report(epoch_records)

    def _save_training_time_report(self, epoch_records: list):
        if not epoch_records:
            return
        df = pd.DataFrame(epoch_records)
        df.loc['total'] = df[['gpu_time_s']].sum()
        df.loc['total', 'epoch'] = 'total'
        df.loc['total', 'avg_loss'] = '-'
        df.loc['total', 'gpu_ms_per_step'] = '-'
        total_s = df['gpu_time_s'].sum()
        out_path = os.path.join(self.args.checkpoint_dir, "training_time.csv")
        df.to_csv(out_path, index=False)
        print(f"Training time report saved to {out_path}")
        print(f"Total GPU training time: {total_s:.2f}s ({total_s/60:.2f}min)")