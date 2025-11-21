import torch
import torch.nn as nn
import torch.nn.functional as F
from model.series_decomp import series_decomp
from model.embed import DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp
from model.encoder import Encoder
from model.decoder import Decoder
from model.MTM import MTM


class Model(nn.Module):
    """
    Main STMNet model for time series forecasting
    Combines series decomposition, MTM module, encoder-decoder architecture
    """

    def __init__(self, args):
        """
        Initialize the STMNet model with configuration parameters

        Args:
            args: Configuration arguments containing model hyperparameters
        """
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.e_layers = args.e_layers
        self.d_layers = args.d_layers

        # Series decomposition module
        self.decomp = series_decomp(kernel_size=args.moving_avg)

        # Embedding layers for encoder and decoder
        self.embed = DataEmbedding_wo_pos(c_in=args.c_in, d_model=args.d_model)
        self.dec_embed = DataEmbedding_wo_pos_temp(c_in=args.c_in, d_model=args.d_model)

        # Encoder and decoder modules
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        # Multi-scale Temporal Memory module
        self.mtm = MTM(feature_dim=args.c_in, d_model=args.d_model, M=args.MTM_factor)

    def forward(self, batch_x, batch_x_mark):
        """
        Forward pass of the STMNet model

        Args:
            batch_x: Input sequence tensor [batch_size, seq_len, features]
            batch_x_mark: Time features tensor [batch_size, seq_len, time_features]

        Returns:
            tuple: (final prediction, hierarchical predictions)
        """
        # Decompose input into seasonal and trend components
        seasonal_dec, trend_dec = self.decomp(batch_x.detach().clone())

        # Initialize trend prediction with mean of input sequence
        mean = torch.mean(batch_x.detach().clone(), dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([batch_x.shape[0], self.pred_len, batch_x.shape[2]], device=batch_x.device)
        trend_init = mean

        # Process input through MTM module if enabled
        if self.args.MTM == True:
            X_hat, T_hat = self.mtm(batch_x)
            embed_enc_in = X_hat
        else:
            embed_enc_in = self.embed(batch_x, batch_x_mark)

        # Encode the input sequence
        enc_out = self.encoder(embed_enc_in)

        # Prepare decoder input based on MTM setting and padding strategy
        if self.args.MTM == True:
            if self.args.seq_len < self.args.pred_len:
                # Handle cases where prediction length exceeds sequence length
                if self.args.padStrategy == 'repeat':
                    embed_dec_in = T_hat
                    embed_dec_in = embed_dec_in.repeat(1, self.args.pred_len // self.args.seq_len + 1, 1)[:,
                                   :self.args.pred_len, :]
                elif self.args.padStrategy == 'mean':
                    trend_dec_pad = torch.mean(T_hat.detach().clone(), dim=1).unsqueeze(1).repeat(1,
                                                                                                  self.args.pred_len - self.args.seq_len,
                                                                                                  1)
                    embed_dec_in = torch.cat([T_hat, trend_dec_pad], dim=1)
                elif self.args.padStrategy == 'zeros':
                    trend_dec_pad = torch.zeros(
                        [T_hat.shape[0], self.args.pred_len - self.args.seq_len, T_hat.shape[2]],
                        device=trend_dec.device)
                    embed_dec_in = torch.cat([T_hat, trend_dec_pad], dim=1)
                else:
                    raise ValueError(
                        f"Invalid padStrategy: '{self.args.padStrategy}'. Expected one of: 'repeat', 'mean', 'zeros'")
            else:
                embed_dec_in = self.dec_embed(trend_dec)
        else:
            if self.args.seq_len < self.args.pred_len:
                # Handle padding for non-MTM mode
                if self.args.padStrategy == 'repeat':
                    embed_dec_in = self.dec_embed(trend_dec)
                    embed_dec_in = embed_dec_in.repeat(1, self.args.pred_len // self.args.seq_len + 1, 1)[:,
                                   :self.args.pred_len, :]
                elif self.args.padStrategy == 'mean':
                    trend_dec_pad = torch.mean(trend_dec.detach().clone(), dim=1).unsqueeze(1).repeat(1,
                                                                                                      self.args.pred_len - self.args.seq_len,
                                                                                                      1)
                    trend_dec = torch.cat([trend_dec, trend_dec_pad], dim=1)
                    embed_dec_in = self.dec_embed(trend_dec)
                elif self.args.padStrategy == 'zeros':
                    trend_dec_pad = torch.zeros(
                        [trend_dec.shape[0], self.args.pred_len - self.args.seq_len, trend_dec.shape[2]],
                        device=trend_dec.device)
                    trend_dec = torch.cat([trend_dec, trend_dec_pad], dim=1)
                    embed_dec_in = self.dec_embed(trend_dec)
                else:
                    raise ValueError(
                        f"Invalid padStrategy: '{self.args.padStrategy}'. Expected one of: 'repeat', 'mean', 'zeros'")
            else:
                embed_dec_in = self.dec_embed(trend_dec)

        # Decode to get seasonal and trend components
        seasonal_part, trend_part, hierarchy_prediction = self.decoder(embed_dec_in, enc_out, trend=trend_init)

        # Combine seasonal and trend components for final prediction
        dec_out = seasonal_part + trend_part

        # Stack hierarchical predictions including final output
        hierarchy_prediction.append(dec_out)
        hierarchy_prediction = torch.stack(hierarchy_prediction, dim=0)

        return dec_out, hierarchy_prediction