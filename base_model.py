# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:22:57 2025

@author: Kévin
"""

import torch
import torch.optim as optim
import torch.nn as nn
import timm
import pytorch_lightning as pl

from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix


class BaseModel(pl.LightningModule):

    def __init__(self, model_name, num_classes=4, learning_rate= 1e-3):

        super().__init__()

        if model_name == "vit_b_16":
          self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
        if model_name == "vit_b_32":
          self.model = timm.create_model("vit_base_patch32_224", pretrained=True, num_classes=num_classes)
        if model_name == "deit_s":
          self.model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=num_classes)
        if model_name == "convnext_t":
          self.model = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.train_accuracy(outputs, labels)

        self.log_dict({'train_loss':loss,"train_acc":acc}, on_step=True,prog_bar=True,logger=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        val_acc = self.val_accuracy(predicted, labels)
        self.log_dict({'val_loss':val_loss,"val_acc":val_acc},prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': val_loss, 'val_acc': val_acc}


    def on_validation_epoch_end(self):
        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        test_acc = self.test_accuracy(predicted, labels)
        self.log_dict({'test_loss':test_loss,"test_acc":test_acc},prog_bar=True, on_step=False, on_epoch=True)
        self.confusion_matrix.update(outputs.argmax(dim=1), labels)
        return {'test_loss': test_loss, 'test_acc': test_acc}

    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        fig_, ax_ = self.confusion_matrix.plot()
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        return optimizer
