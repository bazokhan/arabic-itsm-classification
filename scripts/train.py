"""
CLI training script for Arabic ITSM classification.

Usage:
    python scripts/train.py --task l1 --epochs 5 --lr 2e-5
    python scripts/train.py --tasks l1 l2 l3 --epochs 10 --lr 1e-5
    python scripts/train.py --multi-task --epochs 10 --lr 1e-5
    python scripts/train.py --task l1 --config configs/model_config.yaml

This script wraps the logic in Notebook 04 into a reusable CLI suitable
for running on cloud compute (Colab, Kaggle, cloud VM) or locally.

Output directory naming:
    --task l3             → models/marbert_l3_best/
    --tasks l1 l2 l3      → models/marbert_l1_l2_l3_best/
    --multi-task          → models/marbert_multi_task_best/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MarBERTv2 on Arabic ITSM tickets")
    p.add_argument("--task", default="l1", choices=["l1", "l2", "l3", "priority", "sentiment"],
                   help="Single-task mode (ignored if --multi-task or --tasks is set)")
    p.add_argument("--tasks", nargs="+",
                   choices=["l1", "l2", "l3", "priority", "sentiment"],
                   help="Subset of tasks to train jointly (alternative to --multi-task)")
    p.add_argument("--multi-task", action="store_true",
                   help="Train all 5 tasks (L1, L2, L3, Priority, Sentiment) jointly")
    p.add_argument("--model", default="UBC-NLP/MARBERTv2",
                   help="HuggingFace model ID")
    p.add_argument("--data-dir", default="data/processed",
                   help="Directory with processed train/val/test CSV files")
    p.add_argument("--output-dir", default="models",
                   help="Directory to save checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable mixed-precision training")
    p.add_argument("--config", default=None,
                   help="Path to model_config.yaml (overrides individual args)")
    return p.parse_args()


def main():
    args = parse_args()

    # Lazy imports — only load heavy packages when actually running
    import torch
    import numpy as np
    import pandas as pd
    import pickle
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    import mlflow
    from tqdm.auto import tqdm

    from arabic_itsm.data.preprocessing import ArabicTextNormalizer
    from arabic_itsm.data.dataset import ITSMDataset
    from arabic_itsm.models.classifier import MarBERTClassifier
    from arabic_itsm.utils.metrics import compute_classification_metrics

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    print(f"Device: {device} | FP16: {use_fp16}")

    data_dir = Path(args.data_dir)
    if args.multi_task:
        tasks = ["l1", "l2", "l3", "priority", "sentiment"]
        task_label = "multi_task"
    elif args.tasks:
        tasks = args.tasks
        task_label = "_".join(tasks)
    else:
        tasks = [args.task]
        task_label = args.task

    out_dir = Path(args.output_dir) / f"marbert_{task_label}_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    # Filter encoders for active tasks
    active_encoders = {t: label_encoders[t] for t in tasks}
    num_classes = {t: len(le.classes_) for t, le in active_encoders.items()}
    print(f"Tasks: {tasks} | Classes: {num_classes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    normalizer = ArabicTextNormalizer()

    train_ds = ITSMDataset(train_df, tokenizer, normalizer, active_encoders,
                           args.max_length, tasks=tasks)
    val_ds = ITSMDataset(val_df, tokenizer, normalizer, active_encoders,
                         args.max_length, tasks=tasks)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = MarBERTClassifier(args.model, num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_f1 = 0.0
    mlflow.set_experiment("arabic-itsm-marbert")

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = {f"label_{t}": batch[f"label_{t}"].to(device) for t in tasks}

                with torch.amp.autocast('cuda', enabled=use_fp16):
                    out = model(ids, mask, **labels)
                    loss = out["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Validation
            model.eval()
            all_preds = {t: [] for t in tasks}
            all_labels = {t: [] for t in tasks}
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    out = model(ids, mask)
                    for t in tasks:
                        preds = torch.argmax(out[f"logits_{t}"], dim=-1)
                        all_preds[t].extend(preds.cpu().numpy())
                        all_labels[t].extend(batch[f"label_{t}"].numpy())

            # Compute Metrics
            metrics = {}
            f1_scores = []
            for t in tasks:
                m = compute_classification_metrics(all_labels[t], all_preds[t])
                metrics[f"{t}_macro_f1"] = m["macro_f1"]
                f1_scores.append(m["macro_f1"])
            
            avg_f1 = np.mean(f1_scores)
            metrics["avg_macro_f1"] = avg_f1
            
            print(f"Epoch {epoch} | avg_f1={avg_f1:.4f}")
            for t in tasks:
                print(f"  {t:<10}: {metrics[f'{t}_macro_f1']:.4f}")
                
            mlflow.log_metrics(metrics, step=epoch)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                model.encoder.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
                torch.save(model.heads.state_dict(), out_dir / "heads.pt")
                print(f"  ✓ Checkpoint saved (avg_f1={best_f1:.4f})")

    print(f"\nDone. Best val avg macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {out_dir}")

    # --- Final test-set evaluation (saved for multi-seed aggregation) ---
    print("\nRunning final test-set evaluation...")
    test_df = pd.read_csv(data_dir / "test.csv")
    test_ds = ITSMDataset(test_df, tokenizer, normalizer, active_encoders,
                          args.max_length, tasks=tasks)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    # Reload best checkpoint
    from transformers import AutoModel
    model.heads.load_state_dict(torch.load(out_dir / "heads.pt", map_location=device))
    model.encoder = AutoModel.from_pretrained(str(out_dir)).to(device)
    model.eval()

    test_preds = {t: [] for t in tasks}
    test_labels = {t: [] for t in tasks}
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            for t in tasks:
                preds = torch.argmax(out[f"logits_{t}"], dim=-1)
                test_preds[t].extend(preds.cpu().numpy())
                test_labels[t].extend(batch[f"label_{t}"].numpy())

    test_metrics = {}
    for t in tasks:
        m = compute_classification_metrics(test_labels[t], test_preds[t])
        test_metrics[f"{t}_macro_f1"] = m["macro_f1"]
        test_metrics[f"{t}_accuracy"] = m.get("accuracy", 0.0)
        print(f"  Test {t:<10}: macro_f1={m['macro_f1']:.4f}")

    import json
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
