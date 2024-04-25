
import torch
import torch.nn as nn
import lightning as L
from hyperparameters import HyperParameters
from models.resnet_models import ResNetModel
from sklearn.metrics import cohen_kappa_score
from scheduler import GradualWarmupScheduler


class LightningModel(L.LightningModule):
    def __init__(self, model, h_params: HyperParameters,
                 is_big_image_tile: bool = False):
        # This is where paths and options should be stored. I also store the
        # train_idx, val_idx for cross validation since the dataset are defined
        # in the module !
        super().__init__()

        self.model = model
        # self.hparams.update(h_params)
        self.is_big_image_tile = is_big_image_tile
        self.save_hyperparameters(h_params, ignore=["model"])

        self.validation_step_outputs = []
        self.testing_step_outputs = []

    def forward(self, batch):
        # What to do with a batch in a forward. Usually simple if everything is already defined in the model.
        return self.model(batch['image'])

    def cross_entropy_loss(self, logits, gt):
        # How to calculate the loss. Note this method is actually not a part of pytorch lightning ! It's only good practice
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)
    
    def bce_with_logits_loss(self, logits, target):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, target)

    def configure_optimizers(self):
        # Optimizers and schedulers. Note that each are in lists of equal length to allow multiple optimizers (for GAN for example)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     #  weight_decay=3e-6,
                                     lr=self.hparams.learning_rate / self.hparams.warmup_factor)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.num_epochs - self.hparams.num_warmup_epochs)
        # lr_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, )
        warmup_scheduler = GradualWarmupScheduler(optimizer,
                                                  multiplier=self.hparams.warmup_factor,
                                                  total_epoch=self.hparams.num_warmup_epochs,
                                                  after_scheduler=cosine_scheduler)
        
        # return {"optimizer": optimizer, "lr_scheduler": lr_plateau_scheduler, "monitor": "kappa"}
        return [optimizer], [warmup_scheduler]

    def training_step(self, batch, batch_idx):
        # This is where you must define what happens during a training step (per batch)
        logits = self(batch)
        
        loss = 0
        if self.is_big_image_tile:
            loss = self.bce_with_logits_loss(logits, batch['target'])
        else:
            loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)  # You need to unsqueeze in case you do multi-gpu training
            # predicted_labels = logits.argmax(1)

        # Logging the Training loss
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)

        loss = 0
        predicted_labels = 0
        gt = None

        if self.is_big_image_tile:
            loss = self.bce_with_logits_loss(logits, batch['target'])
            predicted_labels = logits.sigmoid().sum(1).round()
            gt = batch['target'].sum(1)
        else:
            loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)
            predicted_labels = logits.argmax(1)
            gt = batch['target']

        self.validation_step_outputs.append({'val_loss': loss, 
                                             'preds': predicted_labels,
                                             'gt': gt})

        # Logging the Validation loss
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # This is what happens at the end of validation epoch. Usually gathering all predictions
        # outputs is a list of dictionary from each step.
        avg_loss = torch.cat([out['val_loss'].reshape(1)
                              for out in self.validation_step_outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in self.validation_step_outputs], dim=0)
        gt = torch.cat([out['gt'] for out in self.validation_step_outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        # Calculating the Cohen Kappa Score
        kappa = cohen_kappa_score(preds, gt, weights='quadratic')

        # Printing the Validation step outputs to the console
        if self.trainer.global_rank==0:
            print(f'Epoch {self.current_epoch}: val_loss: {avg_loss:.2f}, cohen_kappa_score: {kappa:.4f}')

        # Clearing the validation outputs
        self.validation_step_outputs.clear()

        # Logging the Cohen Kappa Score
        self.log('kappa', kappa, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)

        loss = 0
        predicted_labels = 0
        gt = None

        if self.is_big_image_tile:
            loss = self.bce_with_logits_loss(logits, batch['target'])
            predicted_labels = logits.sigmoid().sum(1).round()
            gt = batch['target'].sum(1)
        else:
            loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)
            predicted_labels = logits.argmax(1)
            gt = batch['target']

        # Creating the resultant dictionary
        result = {'test_loss': loss, 'preds': predicted_labels, 'gt': gt}

        # Adding to the testing outputs
        self.testing_step_outputs.append(result)

        return result

    # def test_step(self, batch, batch_idx):
    #     # This is where you must define what happens during a validation step (per batch)
    #     logits = self(batch)
    #     loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)
    #     preds = logits.argmax(1).detach().cpu().numpy()
    #     gt = batch['target'].detach().cpu().numpy()
    #     kappa = cohen_kappa_score(preds, gt, weights='quadratic')
    #     result = {'test_loss': loss, 'preds': preds, 'gt': gt, 'kappa': kappa}
    #     self.validation_step_outputs.append(result)
    #     return result
    
    def on_test_epoch_end(self):
        output = []
        if self.trainer.is_global_zero:
            outputs = self.all_gather(self.validation_step_outputs)
        # This is what happens at the end of validation epoch. Usually gathering all predictions
        # outputs is a list of dictionary from each step.
        avg_loss = torch.cat([out['test_loss'].reshape(1) for out in self.testing_step_outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in self.testing_step_outputs], dim=0)
        gt = torch.cat([out['gt'] for out in self.testing_step_outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        # Calculating the Cohen Kappa Score
        kappa = cohen_kappa_score(preds, gt, weights='quadratic')

        # Printing the Testing step outputs to the console
        if self.trainer.global_rank==0:
            print(f'Epoch {self.current_epoch}: test_loss: {avg_loss:.2f}, cohen_kappa_score: {kappa:.4f}')

        # Clearing the testing outputs
        self.testing_step_outputs.clear()

    # def on_test_epoch_end(self):
    #     if self.trainer.is_global_zero:
    #         outputs = self.all_gather(self.validation_step_outputs)
    #         return outputs
    #     else:
    #         return self.validation_step_outputs
    
    def predict_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)

        loss = 0
        predicted_labels = 0
        gt = None

        if self.is_big_image_tile:
            loss = self.bce_with_logits_loss(logits, batch['target'])
            predicted_labels = logits.sigmoid().sum(1).round().detach().cpu().numpy()
            gt = batch['target'].sum(1).detach().cpu().numpy()
        else:
            loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)
            predicted_labels = logits.argmax(1).detach().cpu().numpy()
            gt = batch['target'].detach().cpu().numpy()
        
        # loss = self.cross_entropy_loss(logits, batch['target']).unsqueeze(0)
        # preds = logits.argmax(1).detach().cpu().numpy()
        # gt = batch['target'].detach().cpu().numpy()

        # Calculating the Cohen Kappa Score
        kappa = cohen_kappa_score(predicted_labels, gt, weights='quadratic')

        # Creating the resultant dictionary
        result = {'test_loss': loss, 'preds': predicted_labels, 'gt': gt, 'kappa': kappa}

        # # Clearing the testing outputs
        # self.testing_step_outputs.append(result)

        return result
