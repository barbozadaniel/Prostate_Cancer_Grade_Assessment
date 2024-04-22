
import torch
import torch.nn as nn
import lightning as L
from hyperparameters import HyperParameters
from models.resnet_models import ResNetModel
from sklearn.metrics import cohen_kappa_score


class LightningModel(L.LightningModule):
    def __init__(self, model, h_params: HyperParameters):
        # This is where paths and options should be stored. I also store the
        # train_idx, val_idx for cross validation since the dataset are defined
        # in the module !
        super().__init__()

        self.model = model
        self.hparams.update(h_params)
        self.save_hyperparameters(ignore=["model"])
        self.validation_step_outputs = []

    def forward(self, batch):
        # What to do with a batch in a forward. Usually simple if everything is already defined in the model.
        return self.model(batch['image'])

    def cross_entropy_loss(self, logits, gt):
        # How to calculate the loss. Note this method is actually not a part of pytorch lightning ! It's only good practice
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)

    def configure_optimizers(self):
        # Optimizers and schedulers. Note that each are in lists of equal length to allow multiple optimizers (for GAN for example)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=3e-6)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10 * self.hparams.learning_rate, epochs=self.hparams.num_epochs)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs=self.hparams.num_epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, )
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "kappa"}
        return {"optimizer": optimizer}
        # return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # This is where you must define what happens during a training step (per batch)
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup_grade']).unsqueeze(0)  # You need to unsqueeze in case you do multi-gpu training
        predicted_labels = logits.argmax(1)
        # Pytorch lightning will call .backward on what is called 'loss' in output
        # 'log' is reserved for tensorboard and will log everything define in the dictionary
        self.log('train_loss', loss)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup_grade']).unsqueeze(0)
        predicted_labels = logits.argmax(1)
        self.validation_step_outputs.append({'val_loss': loss, 'preds': predicted_labels, 'gt': batch['isup_grade']})
        self.log('val_loss', loss)

        return {'val_loss': loss, 'preds': predicted_labels, 'gt': batch['isup_grade']}

    def on_validation_epoch_end(self):
        # This is what happens at the end of validation epoch. Usually gathering all predictions
        # outputs is a list of dictionary from each step.
        avg_loss = torch.cat([out['val_loss'] for out in self.validation_step_outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in self.validation_step_outputs], dim=0)
        gt = torch.cat([out['gt'] for out in self.validation_step_outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, kappa: {kappa:.4f}')

        self.log('kappa', kappa)
        self.validation_step_outputs.clear()

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup_grade']).unsqueeze(0)
        predicted_labels = logits.argmax(1)
        return {'test_loss': loss, 'preds': predicted_labels, 'gt': batch['isup_grade']}
