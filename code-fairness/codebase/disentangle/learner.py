import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
import codebase.disentangle.models as md

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


class Learner(nn.Module):
    def __init__(self, args, debias_model=None, model=None):
        super().__init__()
        self.args = args
        self.name = self.args.name

        if debias_model is not None:
            self.debias_model = debias_model
        elif args.debias_mode in ['IPM_Embedding', 'Adversarial', 'Fairness']:
            print('experiment_id:{} debias_mode: {}-{} unniform-rate: {} dataset: {} lambda: {} lambdaconfounder: {} is_debias: {} is_deconfound:{}'.format(
                args.experiment_id, args.debias_mode, args.impression_or_click, args.uni_percent, args.dataset,
                args.lambda_impression, args.lambda_confounder, self.args.is_debias, self.args.confounder))
            self.cross_layer = md.DCN(num_cont_fields=args.user_dim, cross_depth=args.cross_depth,
                                      fc_dims=args.fc_dims).to(device)
            if self.args.downstream == 'MLP':
#                 self.model = md.DirectMethod(name=args.debias_mode, args=args).to(device)
                self.model = md.DoubleModelCTR(mode=args.debias_mode,
                                               x_dim=self.args.user_dim,
                                               a_dim=self.args.item_dim,
                                               emb_dim1=self.args.user_emb_dim,
                                               emb_dim2=self.args.item_emb_dim,
                                               layer_dim=self.args.ctr_layer_dims,
                                               y_space=self.args.prediction_size,
                                               x_a_size=self.args.user_item_size,
                                               ctr_classweight=self.args.ctr_classweight,
                                               is_debias=self.args.is_debias).to(device)
                # elif self.args.downstream == 'DCN':
                #     self.mode = md.lightGCN(args)
            else:
                self.model = md.NeuBPR(args=self.args).to(device)
            if self.args.debias_mode == 'Adversarial':
                self.debias_model = md.IPMEmbedding(x_dim=self.args.user_dim,
                                                a_dim=self.args.item_dim,
                                                emb_dim1=self.args.user_emb_dim,
                                                emb_dim2=self.args.item_emb_dim,
                                                layer_dim=self.args.ipm_layer_dims,
                                                item_space=args.prediction_size,
                                                x_a_size=self.args.user_item_size,
                                                confounder_dims=self.args.confounder_dims,
                                                embedding_classweight=self.args.embedding_classweight).to(device)
            elif self.args.debias_mode == 'Fairness':
                self.debias_model = md.FairnessRep(x_dim=self.args.user_dim,
                                                a_dim=self.args.item_dim,
                                                emb_dim1=self.args.user_emb_dim,
                                                emb_dim2=self.args.item_emb_dim,
                                                layer_dim=self.args.ipm_layer_dims,
                                                item_space=args.prediction_size,
                                                x_a_size=self.args.user_item_size,
                                                confounder_dims=self.args.confounder_dims,
                                                embedding_classweight=self.args.embedding_classweight).to(device)
            if self.args.confounder:
                self.confounder = md.ConfounderInference(args).to(device)
        else:
            print('experiment_id:{} debias_mode: {}-{} unniform-rate: {} dataset: {} lambda: {}-{}'.format(
                args.experiment_id, args.debias_mode, args.impression_or_click, args.uni_percent, args.dataset,
                args.lambda_impression, args.lambda_click))

            self.model = md.DoubleModelCTR(mode=args.debias_mode,
                                           x_dim=self.args.user_dim,
                                           a_dim=self.args.item_dim,
                                           emb_dim1=self.args.user_emb_dim,
                                           emb_dim2=self.args.item_emb_dim,
                                           layer_dim=self.args.ctr_layer_dims,
                                           y_space=self.args.prediction_size,
                                           x_a_size=self.args.user_item_size,
                                           ctr_classweight=self.args.ctr_classweight,
                                           is_debias=self.args.is_debias).to(device)
            # self.debias_model = md.IPMEmbedding(x_dim=self.args.user_dim + self.args.fc_dims[-1],
            #                                     a_dim=self.args.item_dim,
            #                                     emb_dim1=self.args.user_emb_dim,
            #                                     emb_dim2=self.args.item_emb_dim,
            #                                     layer_dim=self.args.ipm_layer_dims,
            #                                     item_space=args.prediction_size,
            #                                     x_a_size=self.args.user_item_size,
            #                                     confounder_dims=self.args.confounder_dims,
            #                                     ctr_classweight=self.args.ctr_classweight).to(device)
            print('Unknown Method')

        # if args.debias_mode = 'dev':
        #     self.debias_model = debias_model

    def pretrain(self, x, y, x_bar=None, a=None, savestep=0):
        if savestep % 10 == 0 and savestep > 0:
            ut.save_model_by_name(model_dir=self.args.model_dir, train_mode=self.args.train_mode,
                                  model=self.debias_model, global_step=savestep)
        if a is None:
            if self.args.debias_mode == "DensityRatio":
                return self.debias_model.loss(x, y)
            if self.args.debias_mode == "IPM":
                return self.debias_model.caldistance(x, x_bar)
            if self.args.debias_mode == "IPM_Embedding":
                return self.debias_model.caldistance(x, x_bar)
        else:
            return self.train(x, a, y)

    def learn(self, x, a, r, y, gx, ga, savestep=0, minmax_turn='max'):
        '''
        backward
        :param x:
        :param a:
        :param r:
        :param y:
        :param savestep:
        :param minmax_turn: in max turn, optimazing the parameters in D to maximize discriminator loss. In min turn, minimize parameters except the
        :return:
        '''
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
            if self.args.item_dim == 1:
                a = torch.tensor(a, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.item_dim])
            else:
                a = torch.tensor(a, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.item_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.user_dim])
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.item_dim])
        if self.args.confounder:
            y_true = torch.tensor(y, dtype=torch.int64).to(device)
        gx = torch.tensor(gx, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
        ga = torch.tensor(ga, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
        r = torch.tensor(r, dtype=torch.int64).to(device)
        y = torch.tensor(y, dtype=torch.int64).to(device)


        if not self.args.is_debias:
            ctr_loss = self.model.loss(x.to(device), a.to(device), y.to(device))
            return ctr_loss
        if self.args.debias_mode == 'Fairness':

            x_debias_emb, emb_loss, kl, sensitive_gx, sensitive_ga, sensitive_non  = self.debias_model.discepency(x.to(device), a.to(device), gx.to(device), ga.to(device),
                                                                         self.args.debias_mode, r_true=r)

            if minmax_turn == 'max':
                return emb_loss
            else:
                ctr_loss = self.model.loss(x_debias_emb, a.to(device), y.to(device))
                return self.args.lambda_impression * emb_loss, ctr_loss, self.args.lambda_impression * kl
        else:
            ctr_loss = self.model.loss(x.to(device), a.to(device), y.to(device))
            return ctr_loss
    def get_disentangle(self, x, gx, ga):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.user_dim])

        gx = torch.tensor(gx, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
        ga = torch.tensor(ga, dtype=torch.int64).to(device).reshape([x.size()[0], 1])

        return self.debias_model.disentangle.discepency(x, gx, ga)

    def predict(self, x, a, mode=None):
        '''
        forward
        :param x:
        :param a:
        :param mode:
        :return:
        '''
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
            if self.args.item_dim == 1:
                a = torch.tensor(a, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.item_dim])
            else:
                a = torch.tensor(a, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.item_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.user_dim])
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.item_dim])

        if not self.args.is_debias:
            ctr_loss = self.model.predict(x.to(device), a.to(device))
            return ctr_loss

        if self.args.debias_mode == 'Adversarial':
            if self.args.confounder:
                z, x_emb, _, _ = self.confounder.get_confounder(x.to(device), a.to(device))
                x_debias_emb = self.debias_model.embedding_with_confounder(x_emb, z)
                y = self.model.predict(x_debias_emb.to(device), a.to(device))
            else:
                # x_debias_emb = self.debias_model.embedding(x_dcn_embedding)
                x_debias_emb = self.debias_model.embedding(x.to(device))
                y = self.model.predict(x_debias_emb.to(device), a.to(device))

        elif self.args.debias_mode == 'Fairness':
            x_debias_emb = self.debias_model.get_fair_emb(x.to(device))
            y = self.model.predict(x_debias_emb.to(device), a.to(device))

        elif self.args.debias_mode == "IPM_Embedding":

            x_debias_emb = self.debias_model.embedding(x.to(device))
            y = self.model.predict(x_debias_emb.to(device), a.to(device))

        elif self.args.debias_mode == 'Catboost':
            if mode == 'test_catboost':
                y = self.catboost_model.predict(x.detach().cpu().numpy())
                return y
            x_dcn_embedding = self.cross_layer.embeddings(x.to(device))
            # x_debias_emb = self.debias_model.embedding(x_dcn_embedding)
            x_debias_emb = self.debias_model.embedding(x.to(device))
            y = self.model.predict(x_debias_emb.to(device), a.to(device))

        else:
            x_dcn_embedding = self.cross_layer.embeddings(x.to(device))
            x_debias_emb = self.debias_model.embedding(x_dcn_embedding)
            y = self.model.predict(x.to(device), a.to(device))
        return y