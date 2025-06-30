# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:40:30 2025

@author: Kévin
"""
import torch

from models.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
from models.dylvvit import LVViTDiffPruning, LVViT_Teacher
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher
from losses import ConvNextDistillDiffPruningLoss, DistillDiffPruningLoss_dynamic

import utils

def define_model_teacher(args, sparse_ratio, load_student=False):
    
    if args.model == 'convnext-t':
        model = AdaConvNeXt(sparse_ratio=sparse_ratio, 
                pruning_loc=[1,2,3],
                drop_path_rate=args.drop_path, 
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
                num_classes=args.nb_classes)
        teacher_model = ConvNeXt_Teacher(num_classes=args.nb_classes)
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        
    elif args.model == 'convnext-s':
        model = AdaConvNeXt(sparse_ratio=sparse_ratio, 
                            pruning_loc=[3,6,9],
                            drop_path_rate=args.drop_path, 
                            layer_scale_init_value=args.layer_scale_init_value,
                            head_init_scale=args.head_init_scale,
                            num_classes=args.nb_classes,
                            depths=[3, 3, 27, 3])
        teacher_model = ConvNeXt_Teacher(depths=[3, 3, 27, 3])
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        
    elif args.model == 'convnext-b':
        model = AdaConvNeXt(sparse_ratio=sparse_ratio, 
                            pruning_loc=[3,6,9],
                            drop_path_rate=args.drop_path, 
                            layer_scale_init_value=args.layer_scale_init_value,
                            head_init_scale=args.head_init_scale,
                            num_classes=args.nb_classes,
                            depths=[3, 3, 27, 3], 
                            dims=[128, 256, 512, 1024])
        teacher_model = ConvNeXt_Teacher(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        
    elif args.model == 'lvvit-s':
        PRUNING_LOC = [4,8,12] 
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill,
        )
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = LVViT_Teacher(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True
        )
        
    elif args.model == 'lvvit-m':
        PRUNING_LOC = [5,10,15] 
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill
        )
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = LVViT_Teacher(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True
        )
        
    elif args.model == 'deit-s':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, num_classes = args.nb_classes, kl=args.kl
        )
        pretrained = torch.load(args.pretrained_teacher_model , map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)
        
    elif args.model == 'deit-b' or args.model == 'vit_b_16':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, drop_path_rate=args.drop_path, num_classes = args.nb_classes, kl=args.kl)
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)
    
    elif args.model == 'vit_b_32':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, drop_path_rate=args.drop_path, num_classes = args.nb_classes, kl=args.kl)
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)
        
    elif args.model == 'swin-t':
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[1,2,3], sparse_ratio=sparse_ratio
        )
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = SwinTransformer_Teacher(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7)
        
    elif args.model == 'swin-s':
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=sparse_ratio
        )
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = SwinTransformer_Teacher(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7)
        
    elif args.model == 'swin-b':
        model = AdaSwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=sparse_ratio
        )
        pretrained = torch.load(args.pretrained_teacher_model, map_location='cpu')
        teacher_model = SwinTransformer_Teacher(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7)
    
    
    
    if 'convnext' in args.model or 'deit' in args.model or 'swin' in args.model or 'vit_b' in args.model:
        if 'model' in pretrained:
            pretrained = pretrained['model']
        elif 'state_dict' in pretrained:
            pretrained = pretrained['state_dict']
        pretrained = {k.replace('module.', '').replace('model.', ''): v for k, v in pretrained.items()}
                
    
    utils.load_state_dict(model, pretrained)
    utils.load_state_dict(teacher_model, pretrained)
    teacher_model.eval()
    teacher_model = teacher_model.to(args.device)
    print('success load teacher model weight')
    
    if load_student:
        weights = torch.load(args.pretrained_student_model, map_location='cpu', weights_only=False)['model']
        model.load_state_dict(weights, strict=False)
    
    return model, teacher_model


def define_criterion(args, teacher_model, criterion, sparse_ratio, KEEP_RATE):
    
    if 'convnext' in args.model:
        criterion = ConvNextDistillDiffPruningLoss(
                teacher_model, criterion, clf_weight=1.0, keep_ratio=sparse_ratio, mse_token=True, ratio_weight=10.0, distill=args.distill, kl=args.kl)
    elif 'swin' in args.model:
        criterion = ConvNextDistillDiffPruningLoss(
                teacher_model, criterion, clf_weight=1.0, keep_ratio=sparse_ratio, mse_token=True, ratio_weight=10.0, swin_token=True, distill=args.distill, kl=args.kl)
    elif 'lvvit' in args.model:
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=False, ratio_weight=2.0, distill_weight=0.5, distill=args.distill, kl=args.kl
        )
    elif 'deit' in args.model or 'vit_b' in args.model:
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=True, ratio_weight=args.ratio_weight, distill_weight=0.5, distill=args.distill, kl=args.kl
        )
    
    return criterion