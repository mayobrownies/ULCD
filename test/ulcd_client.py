import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from . import config


class ULCDClient:
    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if config.USE_AMP else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)


    def compute_prototypes(self):
        self.model.eval()
        device = next(self.model.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                consensus_latents = self.model.get_consensus_features(x)

                if torch.isnan(consensus_latents).all():
                    continue

                for i in range(len(consensus_latents)):
                    if torch.isnan(consensus_latents[i]).any():
                        continue
                    label = y[i].item()
                    class_latents[label].append(consensus_latents[i].cpu())
                    class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id]:
                class_stack = torch.stack(class_latents[class_id])
                proto = class_stack.mean(dim=0)

                proto_norm = torch.norm(proto)
                if proto_norm > 1e-8 and not torch.isnan(proto_norm):
                    proto = proto / (proto_norm + 1e-8)
                    if not torch.isnan(proto).any():
                        class_prototypes[class_id] = proto
                    else:
                        print(f"Client {self.client_id}: WARNING - NaN prototype for class {class_id} after normalization, skipping")
                else:
                    print(f"Client {self.client_id}: WARNING - Zero-norm prototype for class {class_id} (norm={proto_norm:.6f}), skipping")

        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask

    def compute_contrastive_consensus_loss(self, x_align, y_align, server_prototypes):
        if not server_prototypes or len(server_prototypes) < 2:
            return torch.tensor(0.0).cuda()

        consensus_feats = self.model.get_consensus_features(x_align)

        if torch.isnan(consensus_feats).any() or torch.isinf(consensus_feats).any():
            return torch.tensor(0.0).cuda()

        tau = config.CONTRASTIVE_TEMP

        all_proto_labels = sorted(server_prototypes.keys())
        all_protos = torch.stack([server_prototypes[k].to(device="cuda", dtype=consensus_feats.dtype) for k in all_proto_labels])

        label_to_idx = {label: idx for idx, label in enumerate(all_proto_labels)}
        valid_mask = torch.tensor([label.item() in label_to_idx for label in y_align], device='cuda')

        if not valid_mask.any():
            return torch.tensor(0.0).cuda()

        valid_labels = y_align[valid_mask]
        valid_feats = consensus_feats[valid_mask]

        if torch.isnan(all_protos).any() or torch.isinf(all_protos).any():
            return torch.tensor(0.0).cuda()

        similarities = F.cosine_similarity(valid_feats.unsqueeze(1), all_protos.unsqueeze(0), dim=2) / tau

        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            return torch.tensor(0.0).cuda()

        pos_indices = torch.tensor([label_to_idx[label.item()] for label in valid_labels], device='cuda')

        pos_sim = similarities[torch.arange(len(valid_labels), device='cuda'), pos_indices]

        pos_exp = torch.exp(pos_sim)
        all_exp = torch.exp(similarities).sum(dim=1)

        loss = -torch.log(pos_exp / all_exp + 1e-8).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0).cuda()

        return loss

    def train(self, epochs, server_prototypes=None, round_num=1):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        if config.ULCD_USE_WARMUP:
            proto_weight = min(config.PROTOTYPE_WARMUP_RATE * round_num, config.PROTOTYPE_WEIGHT)
        else:
            proto_weight = config.PROTOTYPE_WEIGHT

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"Client {self.client_id}: LR = {current_lr:.6f}, Proto Weight = {proto_weight:.2f}")

        if config.ULCD_USE_PUBLIC_ALIGNMENT:
            alignment_iter = iter(self.public_loader)
        else:
            alignment_iter = iter(self.train_loader)

        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_proto_loss = 0
            total_contrastive_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                with autocast(enabled=config.USE_AMP):
                    out = self.model(x)
                    ce_loss = self.ce_loss(out, y)
                    loss = ce_loss
                    total_ce_loss += ce_loss.item()

                    if server_prototypes is not None:
                        try:
                            x_align, y_align = next(alignment_iter)
                        except StopIteration:
                            if config.ULCD_USE_PUBLIC_ALIGNMENT:
                                alignment_iter = iter(self.public_loader)
                            else:
                                alignment_iter = iter(self.train_loader)
                            x_align, y_align = next(alignment_iter)

                        x_align, y_align = x_align.cuda(), y_align.cuda()
                        feats_align = self.model.get_consensus_features(x_align)

                        if torch.isnan(feats_align).any() or torch.isinf(feats_align).any():
                            print(f"Client {self.client_id}: WARNING - Invalid features detected, skipping alignment")
                            continue

                        mask = torch.tensor([label.item() in server_prototypes for label in y_align], device='cuda')
                        if mask.any():
                            valid_labels = y_align[mask]
                            valid_feats = feats_align[mask]

                            proto_stack = torch.stack([server_prototypes[label.item()].to(device="cuda", dtype=valid_feats.dtype) for label in valid_labels])

                            if torch.isnan(proto_stack).any() or torch.isinf(proto_stack).any():
                                print(f"Client {self.client_id}: WARNING - Invalid server prototypes, skipping alignment")
                            else:
                                sims = F.cosine_similarity(valid_feats, proto_stack, dim=1)
                                proto_align = (1 - sims).mean()

                                if not torch.isnan(proto_align) and not torch.isinf(proto_align):
                                    loss += proto_weight * proto_align
                                    total_proto_loss += proto_align.item()
                                else:
                                    print(f"Client {self.client_id}: WARNING - Invalid proto_align loss")

                        if config.ULCD_USE_CONTRASTIVE:
                            contrastive_loss = self.compute_contrastive_consensus_loss(
                                x_align, y_align, server_prototypes
                            )
                            loss += config.CONTRASTIVE_WEIGHT * contrastive_loss
                            total_contrastive_loss += contrastive_loss.item()

                self.optimizer.zero_grad()

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Client {self.client_id}: WARNING - Invalid loss={loss.item()}, skipping batch")
                    continue

                if config.USE_AMP:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                params_valid = all(
                    not torch.isnan(p).any() and not torch.isinf(p).any()
                    for p in self.model.parameters()
                )
                if not params_valid:
                    print(f"Client {self.client_id}: ERROR - Model parameters became NaN/Inf after update!")
                    print(f"Last loss: CE={ce_loss.item():.4f}, Proto={proto_align.item() if 'proto_align' in locals() else 0:.4f}")
                    raise RuntimeError("Model parameters became NaN/Inf")

                total_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                avg_ce = total_ce_loss / num_batches
                avg_proto = total_proto_loss / num_batches if server_prototypes else 0
                avg_contrastive = total_contrastive_loss / num_batches if server_prototypes and config.ULCD_USE_CONTRASTIVE else 0

                print(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, Proto={avg_proto:.4f}, Contr={avg_contrastive:.4f})")
            else:
                print(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: WARNING - All batches skipped (NaN)")

        if server_prototypes and config.ULCD_PUBLIC_ALIGNMENT_EPOCHS > 0:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.consensus_projection.parameters():
                param.requires_grad = True

            from torch.utils.data import DataLoader
            public_align_loader = DataLoader(
                self.public_loader.dataset,
                batch_size=config.ULCD_PUBLIC_ALIGNMENT_BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            pub_optimizer = torch.optim.Adam(
                self.model.consensus_projection.parameters(),
                lr=config.ULCD_PUBLIC_ALIGNMENT_LR
            )

            for pub_epoch in range(config.ULCD_PUBLIC_ALIGNMENT_EPOCHS):
                for x_pub, y_pub in public_align_loader:
                    x_pub, y_pub = x_pub.cuda(), y_pub.cuda()

                    contrastive_loss = self.compute_contrastive_consensus_loss(
                        x_pub, y_pub, server_prototypes
                    )

                    feats_pub = self.model.get_consensus_features(x_pub)
                    all_proto_labels = sorted(server_prototypes.keys())
                    label_to_idx = {label: idx for idx, label in enumerate(all_proto_labels)}
                    valid_mask = torch.tensor([label.item() in label_to_idx for label in y_pub], device='cuda')

                    proto_loss = torch.tensor(0.0).cuda()
                    if valid_mask.any():
                        valid_labels = y_pub[valid_mask]
                        valid_feats = feats_pub[valid_mask]
                        proto_stack = torch.stack([server_prototypes[label.item()].cuda() for label in valid_labels])
                        sims = F.cosine_similarity(valid_feats, proto_stack, dim=1)
                        proto_loss = (1 - sims).mean()

                    loss = config.CONTRASTIVE_WEIGHT * contrastive_loss + config.PROTOTYPE_WEIGHT * proto_loss

                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                        continue

                    pub_optimizer.zero_grad()
                    loss.backward()
                    pub_optimizer.step()

            for param in self.model.parameters():
                param.requires_grad = True
