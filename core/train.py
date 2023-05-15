import torch
from tools import MultiItemAverageMeter

def train(base, loaders, config):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for i, (input1_0, input1_1, input2, label1, label2) in enumerate(loader):
        rgb_imgs1, rgb_imgs2, rgb_pids = input1_0, input1_1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs1, rgb_imgs2, rgb_pids = rgb_imgs1.to(base.device),  rgb_imgs2.to(base.device), \
                                        rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        if config.module == 'B':
            rgb_imgs = torch.cat([rgb_imgs1, rgb_imgs2], dim=0)
            rgb_features_map = base.rgb_model(rgb_imgs)
            ir_features_map = base.ir_model(ir_imgs)

            features_map = torch.cat([rgb_features_map, ir_features_map], dim=0)

            features_map = base.shared_model(features_map)

            features, cls_score = base.classifier(features_map)

            pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

            ide_loss = base.pid_creiteron(cls_score, pids)

            total_loss = ide_loss

            base.rgb_model_optimizer.zero_grad()
            base.ir_model_optimizer.zero_grad()
            base.shared_model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.rgb_model_optimizer.step()
            base.ir_model_optimizer.step()
            base.shared_model_optimizer.step()
            base.classifier_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
            })

        elif config.module == 'B_tri':
            rgb_imgs = torch.cat([rgb_imgs1, rgb_imgs2], dim=0)
            rgb_features_map = base.rgb_model(rgb_imgs)
            ir_features_map = base.ir_model(ir_imgs)

            features_map = torch.cat([rgb_features_map, ir_features_map], dim=0)

            features_map = base.shared_model(features_map)

            features, cls_score = base.classifier(features_map)

            pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

            ide_loss = base.pid_creiteron(cls_score, pids)
            tri_loss = base.tri_creiteron(features.squeeze(), pids)

            total_loss = ide_loss + tri_loss

            base.rgb_model_optimizer.zero_grad()
            base.ir_model_optimizer.zero_grad()
            base.shared_model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.rgb_model_optimizer.step()
            base.ir_model_optimizer.step()
            base.shared_model_optimizer.step()
            base.classifier_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'tri_loss': tri_loss.data,
            })

    return meter.get_val(), meter.get_str()







