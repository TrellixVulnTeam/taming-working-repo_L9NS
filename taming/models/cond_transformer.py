import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import einops

from main import instantiate_from_config
from taming.modules.util import SOSProvider,CodeGPT_SOSProvider


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        #print('init_cond_stage_from_ckpt called=============')
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        #print('a_indices.shape: ',a_indices.shape)
        #print('c_indices.shape: ',c_indices.shape)
        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        #print('Forward step, cz_indices=',cz_indices)
        #print('Max Index = ', cz_indices.max())

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        #print('logits.shape: ',logits.shape)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            #print('Before loop in sample function')
            #print('Steps= ',steps)
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                #print('logits.shape: ', logits.shape)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        #print('-------------------------------------------------------')
        #print('min_encoding_indices shape:',info[2].shape)
        #print('-------------------------------------------------------')
        indices = info[2].view(quant_z.shape[0], -1) #info[2]=min_encoding_indices (indices of closest embedding in codebook)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        #print('Decoding Image. bhwc=',bhwc)
        #print('index.shape ', index.shape)
        #print('index.reshape(-1) shape ',index.reshape(-1).shape)
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):

        #print('=====log_images called=====')
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        #print('Creating half sample=========')
        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        #print('z_start_indices: ', z_start_indices)

        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else min(100,self.transformer.config.vocab_size),
                                   callback=callback if callback is not None else lambda k: None)

        #print('index_sample: ', index_sample)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)
        #print('Half sample created==========')

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else min(100,self.transformer.config.vocab_size),
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        #print('=====================shared_step===============')
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.transformer.cross_attention:
            no_decay.add('context_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer

class RVQTransformer (Net2NetTransformer):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 z_codebook_level=0,
                 cond_on_prev_level=False,
                 joint_training=False,
                 end_to_end_sampling=False,
                 ):

        #Adjust vocab size of transformer to that of the chosen codebook level
        self.vocab_size=first_stage_config.params.n_embeds[z_codebook_level]
        transformer_config.params.vocab_size=self.vocab_size

        self.first_stage_config=first_stage_config
        self.transformer_config=transformer_config

        self.cond_on_prev_level = cond_on_prev_level
        self.joint_training = joint_training
        self.end_to_end_sampling = end_to_end_sampling
        self.z_codebook_level = z_codebook_level
        self.c_codebook_level = z_codebook_level-1 if z_codebook_level>0 else None


        print('Constructing RVQTransformer object. transformer_config: ', transformer_config)
        super().__init__(
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config,
                 ckpt_path,
                 ignore_keys,
                 first_stage_key,
                 cond_stage_key,
                 downsample_cond_size,
                 pkeep,
                 sos_token,
                 unconditional)

        if self.joint_training:
            self.sos_cond_stage = CodeGPT_SOSProvider(self.sos_token)
        self.use_sos_cond = False

        #print('self.device=',self.device)
        #self.transformer_config.params.device=self.device

    def get_down_factor(self):
        ch_mult = self.first_stage_config.params.ddconfig.ch_mult
        down_f = 2**(len(ch_mult)-1)
        return down_f

    def get_hidden_dim(self):
        down_f=self.get_down_factor()
        res = self.first_stage_config.params.ddconfig.resolution
        assert res%down_f==0
        return res//down_f

    def decode_to_img(self, index, zshape):
        #print('quant_pred mean, std:',quant_pred.mean())
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        #print('Decoding Image. bhwc=',bhwc)
        #print('index.shape ', index.shape)
        #print('index.reshape(-1) shape ',index.reshape(-1).shape)
        #print('self.first_stage_model.quantizers: ', self.first_stage_model.quantizers)
        #print(self.first_stage_model.quantizers[self.z_codebook_level])
        quant_z = self.first_stage_model.quantizers[self.z_codebook_level].get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    def quant_c_and_ind_to_next_cblvl(self, quant_c, pred_indices, target_shape_bchw):
        pred_indices = self.permuter(pred_indices, reverse=True)
        bhwc = (target_shape_bchw[0],target_shape_bchw[2],target_shape_bchw[3],target_shape_bchw[1])

        quant_pred = self.first_stage_model.quantizers[self.z_codebook_level].get_codebook_entry(
                pred_indices.reshape(-1), shape=bhwc)
        quant_c = einops.rearrange(quant_c,'b (h w) c -> b c h w', h=bhwc[1], w=bhwc[2])
        if self.be_unconditional:
            quant_sum = quant_pred
        else:
            quant_sum = quant_c + quant_pred

        return quant_sum

    def cond_and_pred_to_img(self, quant_c, pred_indices, target_shape_bchw):
        quant_sum = self.quant_c_and_ind_to_next_cblvl(quant_c, pred_indices, target_shape_bchw)
        img = self.first_stage_model.decode(quant_sum)
        return img

    def get_zlevel_one_hot(self,device):
        z_level_tensor=torch.ones((1),dtype=torch.long,device=device)*self.z_codebook_level
        return torch.nn.functional.one_hot(z_level_tensor, num_classes=self.first_stage_config.params.n_levels).float()

    def append_level_encoding(self, x):
        level_encoding=self.get_zlevel_one_hot(quant_z.device)
        print(level_encoding)
        level_encoding=level_encoding.expand(x.shape[0],x.shape[1],-1)
        print(level_encoding.shape)
        quant_z = torch.cat((quant_z,level_encoding), dim=2)
        print(quant_z.shape)


    @torch.no_grad()
    def encode_to_z(self, x, codebook_level=None):
        if codebook_level is None:
            codebook_level=self.z_codebook_level

        #print('------------------------encode_to_z called-------------------------------')
        #print('x.shape:',x.shape)
        #print(self.z_codebook_level)
        #print(self.first_stage_model.quantizers[self.z_codebook_level])
        pre_quant = self.first_stage_model.encode_to_prequant(x)
        #print('pre_quant.shape:',pre_quant.shape)
        all_quantized, all_indices, all_losses = self.first_stage_model.make_quantizations(pre_quant)
        #print(len(all_indices),all_indices[0].shape)
        #print(len(all_quantized),all_quantized[0].shape)
        #print('-------------------------------------------------------')
        if codebook_level>0:
            quant_z = all_quantized[codebook_level] - all_quantized[codebook_level-1]

        else:
            quant_z=all_quantized[codebook_level]

        indices=all_indices[codebook_level]
        indices = indices.view(quant_z.shape[0], -1) 

        #quant_z, _, info = self.first_stage_model.encode(x)
        #indices = info[2].view(quant_z.shape[0], -1) #info[2]=min_encoding_indices (indices of closest embedding in codebook)
        indices = self.permuter(indices)
        #print('encode_to_z done')
        bchw = quant_z.shape
        quant_z = einops.rearrange(quant_z, 'b c h w -> b (h w) c')
        return quant_z, indices, bchw

    @torch.no_grad()
    def encode_to_c(self, c, codebook_level=None):
        if codebook_level is None:
            codebook_level=self.c_codebook_level
        #print('=====encode_to_c called=====')
        #print('c.shape:',c.shape)
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        if self.be_unconditional:
            if self.use_sos_cond:
                #print('======generating SOS for unconditional case========')
                #print('c.shape:',c.shape)
                hidden_dim=self.get_hidden_dim()
                #print('get_hidden_dim:',hidden_dim)

                #Then replace them with sos tokens for dummy cond sequence
                bs=c.shape[0]
                quant_c, _, [_,_,indices] = self.sos_cond_stage.encode(c,bchw=(bs,self.first_stage_config.params.embed_dim,hidden_dim,hidden_dim))

            else:
                quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
            #print('quant_c.shape:',quant_c.shape)
            bchw = quant_c.shape
            quant_c = einops.rearrange(quant_c, 'b c h w -> b (h w) c')
        else:
            pre_quant = self.cond_stage_model.encode_to_prequant(c)
            all_quantized, all_indices, all_losses = self.cond_stage_model.make_quantizations(pre_quant)
            #print('all_quantized[:self.c_codebook_level+1].shape:', all_quantized[:self.c_codebook_level+1].shape)
            #The quantizations in all_quantized are already cumulative over the codebook levels
            quant_c = all_quantized[codebook_level]
            #print('quant_c.shape:',quant_c.shape)
            #print('all_indices len: ',len(all_indices))
            indices=all_indices[codebook_level]
            #print('indices.shape:', indices.shape)
            #indices = einops.rearrange(indices, '(b h w) -> b (h w)') (?)
            indices = indices.view(quant_c.shape[0], -1) 
            bchw = quant_c.shape
            quant_c = einops.rearrange(quant_c, 'b c h w -> b (h w) c')
            #print('reshaped: quant_c.shape:',quant_c.shape)
            #quant_c = quant_c.view(quant_c.shape[0],-1)
        #quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        #print('encode_to_c done')
        return quant_c, indices, bchw

    #TODO: Change this to take preceding codebook level as conditioning stage
    def init_cond_stage_from_ckpt(self, config):
        #print('init_cond_stage_from_ckpt called=============')

        if (config == "__previous_codebook_level__" or self.cond_on_prev_level) and self.z_codebook_level==0:
                print('First stage has lowest codebook level. Proceeding without cond stage.')
                self.be_unconditional=True
                self.cond_on_prev_level=False

        if (config == "__previous_codebook_level__" or self.cond_on_prev_level) and self.z_codebook_level>0:
                print('First stage codebook level is {}. Using codebook_level {} as cond stage'.format(self.z_codebook_level,self.c_codebook_level))
                #model = instantiate_from_config(self.first_stage_config)
                #model = model.eval()
                #model.train = disabled_train
                #self.cond_stage_model = model
                self.cond_stage_model = self.first_stage_model
                self.cond_stage_key = self.first_stage_key

        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key

            if self.transformer_config.target=='taming.modules.transformer.mingpt.CodeGPT':
                self.cond_stage_model = CodeGPT_SOSProvider(self.sos_token)
            else:
                self.cond_stage_model = SOSProvider(self.sos_token)

        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

            
    def forward(self, x, c):
        # one step to produce the logits
        #print('forward, x.shape:',x.shape)
        #print('forward, c.shape:',c.shape)
        z_quant, z_indices, _ = self.encode_to_z(x)
        c_quant, c_indices, _ = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices


        #print('---------concatenating to cz_quants:')
        #print('c_quant.shape:',c_quant.shape)
        #print('z_quant.shape:',z_quant.shape)

        #TODO: Adapt joint training for unequal level sizes
        if self.joint_training:
            zlevel_encoding=self.get_zlevel_one_hot(x.device)
            #zlevel_encoding=zlevel_encoding.expand(x.shape[0],x.shape[1],-1)
        else:
            zlevel_encoding=None

        if not self.transformer.cross_attention:
            if self.cond_on_prev_level:
                cz_quants = torch.cat((c_quant, z_quant), dim=1)
            #TODO: Adjust unconditional case to work with CodeGPT instead of GPT as well!
            else:
                #print('a_indices.shape: ',a_indices.shape)
                #print('c_indices.shape: ',c_indices.shape)
                cz_indices = torch.cat((c_indices, a_indices), dim=1)
                #print('Forward step, cz_indices=',cz_indices)
                #print('Max Index = ', cz_indices.max())

            if self.cond_on_prev_level: 
                target = z_indices
                #print('target.shape: ',target.shape)
                #print('cz_quants.shape:', cz_quants.shape)

                logits, _ = self.transformer(cz_quants[:,:-1,:], zlevel_one_hot=zlevel_encoding)
                #print('logits.shape: ',logits.shape)

                # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
                logits = logits[:, c_quant.shape[1]-1:]

                #print('logits.shape: ',logits.shape)
            else:
                # target includes all sequence elements (no need to handle first one
                # differently because we are conditioning)
                target = z_indices
                #print('target.shape: ',target.shape)
                # make the prediction
                #print('cz_indices.shape:', cz_indices.shape)

                logits, _ = self.transformer(cz_indices[:, :-1],zlevel_one_hot=zlevel_encoding)

                # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
                logits = logits[:, c_indices.shape[1]-1:]

        else:
            if self.cond_on_prev_level: 
                target = z_indices[:,:-1]
                #print('====Before forward')
                #print('z_quant.shape: ',z_quant.shape)
                #print('c_quant.shape: ',c_quant.shape)
                logits, _ = self.transformer(z_quant[:,:-1,:], context = c_quant,zlevel_one_hot=zlevel_encoding)
                #print('logits.shape: ',logits.shape)
            else:
                target = z_indices
                logits, _ = self.transformer(z_indices[:, :-1], context = a_indices,zlevel_one_hot=zlevel_encoding)


        return logits, target

    #TODO: Loss Function for CodeGPT version with vector input!
    def shared_step(self, batch, batch_idx):
        #print('=====================shared_step===============')

        if self.joint_training:
            self.z_codebook_level = 0
            self.c_codebook_level = None
            self.be_unconditional = True
            self.use_sos_cond = True

            x, c = self.get_xc(batch, self.use_sos_cond)
            logits, target = self(x, c)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            self.be_unconditional = False
            self.use_sos_cond = False

            for z_lvl in range(1,self.first_stage_model.n_levels):
                #print('==========loop, z_lvl=', z_lvl)
                self.z_codebook_level = z_lvl
                self.c_codebook_level = z_lvl-1

                x, c = self.get_xc(batch)
                logits, target = self(x, c)
                #print(logits)
                #print(target.max())
                loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        else:
            x, c = self.get_xc(batch)
            logits, target = self(x, c)
            #print('Calculating loss function. logits.shape={}, target.shape={}'.format(logits.shape,target.shape))
            #print(logits.reshape(-1, logits.size(-1)).shape) 
            #print(target.reshape(-1).shape)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        return loss


    @torch.no_grad()
    def log_images_one_lvl(self, batch, return_quantized_sample=False, temperature=None, top_k=None, callback=None, lr_interface=False):

        #print('=====log_images called=====')
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices, z_bchw = self.encode_to_z(x)
        quant_c, c_indices, c_bchw = self.encode_to_c(c)

        #TODO: Adapt half picture sampling to work with CodeGPT
        if not self.cond_on_prev_level and not self.joint_training:
            #print('Creating half sample=========')
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            #print('z_start_indices: ', z_start_indices)

            index_sample = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1]-z_start_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else min(100,self.transformer.config.vocab_size),
                                       callback=callback if callback is not None else lambda k: None)

            #print('index_sample: ', index_sample)
            x_sample = self.decode_to_img(index_sample, z_bchw)
            #print('Half sample created==========')

        #print('------------Sampling-------------')
        #print('quant_z.shape:,', quant_z.shape)
        #print('quant_c.shape:,', quant_c.shape)

        #print('weights: ',self.transformer.tok_emb.weight)
        # sample
        if not self.cond_on_prev_level and not self.joint_training:
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else min(100,self.transformer.config.vocab_size),
                                       callback=callback if callback is not None else lambda k: None)

            x_sample_nopix = self.decode_to_img(index_sample, z_bchw)

        else:
            #print('===========normal sampling==========\n')
            #print('quant_c.shape:',quant_c.shape)
            z_start_quants=quant_z[:, :0, :]
            index_sample = self.sample_codegpt(z_start_quants, quant_c,
                                       steps=quant_z.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else min(100,self.transformer.config.vocab_size),
                                       callback=callback if callback is not None else lambda k: None)
            #x_sample_nopix = self.decode_to_img(index_sample, z_bchw)

            #print('===================Before cond_and_pred_to_img')
            #print('quant_c.shape:',quant_c.shape)
            #print('index_sample.shape:',index_sample.shape)
            if self.be_unconditional:
                dummy_c = self.sos_token*torch.ones(z_bchw[0],z_bchw[2]*z_bchw[3],z_bchw[1],device=quant_c.device,dtype=quant_c.dtype)
                #print('dummy_c.shape:',dummy_c.shape)
                x_sample_nopix_sum = self.cond_and_pred_to_img(dummy_c, index_sample, z_bchw)
            else:
                x_sample_nopix_sum = self.cond_and_pred_to_img(quant_c, index_sample, z_bchw)


        # det sample
        z_start_indices = z_indices[:, :0]

        if not self.cond_on_prev_level and not self.joint_training:
            index_sample_det = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1],
                                       sample=False,
                                       callback=callback if callback is not None else lambda k: None)
            x_sample_det = self.decode_to_img(index_sample_det, z_bchw)

        #else:
            #z_start_quants=quant_z[:, :0, :]
            #index_sample_det = self.sample_codegpt(z_start_quants, quant_c,
            #                           steps=z_indices.shape[1],
            #                           sample=False,
            #                           callback=callback if callback is not None else lambda k: None)


            if self.be_unconditional:
                dummy_c = self.sos_token*torch.ones(z_bchw[0],z_bchw[2]*z_bchw[3],z_bchw[1],device=quant_c.device,dtype=quant_c.dtype)
                x_sample_nopix_sum = self.cond_and_pred_to_img(dummy_c, index_sample, z_bchw)
                #x_sample_det = self.cond_and_pred_to_img(dummy_c, index_sample_det, z_bchw)
            else:
                x_sample_nopix_sum = self.cond_and_pred_to_img(quant_c, index_sample, z_bchw)
                #x_sample_det = self.cond_and_pred_to_img(quant_c, index_sample_det, z_bchw)

        # reconstruction
        #x_rec = self.decode_to_img(z_indices, z_bchw)

        log_resvq = self.first_stage_model.log_images(batch)
        log["inputs"] = x
        log["reconstructions_target_lvl_{}".format(self.z_codebook_level)] = log_resvq["reconstructions_{}".format(self.z_codebook_level)]

        if self.cond_on_prev_level and not self.joint_training:
            log["reconstructions_cond_lvl_{}".format(self.c_codebook_level)] = log_resvq["reconstructions_{}".format(self.c_codebook_level)]
            #log["reconstructions_only_lvl_{}".format(self.z_codebook_level)] = x_rec


        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        #print('Logging samples:===================')
        #print('x_sample_nopix.shape:', x_sample_nopix.shape)
        #print('x_sample_det.shape:', x_sample_det.shape)

        if self.cond_on_prev_level or self.joint_training:
            log["samples_nopix_lvl{}".format(self.z_codebook_level)] = x_sample_nopix_sum
            #log["samples_det_sum_cond"] = x_sample_det_sum
            #log["samples_nopix_only_residual"] = x_sample_nopix
            #log["samples_det_only_residual"] = x_sample_det
        else:
            log["samples_nopix"] = x_sample_nopix
            log["samples_det"] = x_sample_det
            log["samples_half"] = x_sample

        if return_quantized_sample:
            #print('returning quantized sample\n')
            #print('quant_c.shape:',quant_c.shape)
            quant_sample = self.quant_c_and_ind_to_next_cblvl(quant_c, index_sample, z_bchw)
            #print('quant_sample.shape:',quant_sample.shape)
            return log, quant_sample
        else:
            return log


    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        print('==========Starting Image Logging=========')
        #print('head.weight(decoder):',self.transformer.head.weight)
        #print('tok_emb.weight: ',self.transformer.tok_emb.weight)

        if not self.joint_training:
            logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface)

            return logs

        else:
            #print('tok_emb.weight: ',self.transformer.tok_emb.weight)
            self.z_codebook_level = 0
            self.c_codebook_level = None
            self.be_unconditional = True
            self.use_sos_cond = True
            self.cond_on_prev_level=False

            print('==========Logging level 0')
            #print('tok_emb.weight: ',self.transformer.tok_emb.weight)

            if not self.end_to_end_sampling:
                logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface)

            else:
                #Lvl0 is identical for end-to-end sampling
                logs, quant_sample = self.log_images_one_lvl(batch, return_quantized_sample=True, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface)
                logs['sample_end2end_lvl0'] = logs['samples_nopix_lvl0']
                print('quant_sample.shape:',quant_sample.shape)

            self.be_unconditional = False
            self.use_sos_cond = False
            self.cond_on_prev_level=True

            for z_lvl in range(1,self.first_stage_model.n_levels):
                print('==========loop, z_lvl=', z_lvl)
                #print('tok_emb.weight: ',self.transformer.tok_emb.weight)
                self.z_codebook_level = z_lvl
                self.c_codebook_level = z_lvl-1

                #Always sample from ground truth to next level
                new_logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface)

                #Optionally from last level sample to next level
                if self.end_to_end_sampling:
                    z_bchw = quant_sample.shape
                    quant_sample = einops.rearrange(quant_sample,'b c h w -> b (h w) c')
                    z_start_quants = quant_sample[:, :0, :]
                    #Sample new residual indices, using last quantized sample as conditioning

                    #print('===========end2end sampling==========\n')
                    #print('quant_sample.shape:',quant_sample.shape)
                    new_sample_ind = self.sample_codegpt(z_start_quants, c=quant_sample,
                                               steps=quant_sample.shape[1],
                                               sample=True)

                    #Add up predictions and last level quants to new quantized sample
                    quant_sample = self.quant_c_and_ind_to_next_cblvl(quant_sample, new_sample_ind, target_shape_bchw=z_bchw)
                    x_sample_end2end = self.first_stage_model.decode(quant_sample)

                    new_logs['sample_end2end_lvl{}'.format(self.z_codebook_level)]=x_sample_end2end


                logs = {**logs, **new_logs}

            return logs


    def sample_codegpt(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):

        if self.transformer.cross_attention:
            sos = torch.ones((x.shape[0],1,x.shape[2]),dtype=x.dtype, device=x.device) * self.sos_token
            x = torch.cat((sos,x),dim=1)
        else:
            #print('sample_codegpt========================================')
            #print('x.shape:',x.shape)
            #print('c.shape:',c.shape)
            x = torch.cat((c,x),dim=1)
            assert torch.isnan(x).sum()==0

        if self.joint_training:
            zlevel_encoding=self.get_zlevel_one_hot(x.device)
        else:
            zlevel_encoding=None

        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        #print('Before loop in sample function')
        #print('Steps= ',steps)
        x_indices = torch.zeros((x.shape[0],0),device=x.device,dtype=torch.long)
        for k in range(steps):
            assert torch.isnan(x).sum()==0
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            #x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

            if self.transformer.cross_attention:
                #print('In sample loop, x.shape: ',x.shape)
                #print('In sample loop, c.shape: ',c.shape)
                logits, _ = self.transformer(x, context=c, zlevel_one_hot=zlevel_encoding)
            else:
                #print('Sample, x=',x)
                #print('Sample, x.shape=',x.shape)
                logits, _ = self.transformer(x, zlevel_one_hot=zlevel_encoding)
                #print('Sample, logits=', logits)

            #print('in sample, logits.shape: ', logits.shape)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
           # print('probs:',probs)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            #print('ix.shape:',ix.shape)
            #print('x_indices.shape:', x_indices.shape)
            x_indices=torch.cat((x_indices,ix), dim=1)
            bhwc=(ix.shape[0],1,1,x.shape[2])
            ix_quant = self.first_stage_model.quantizers[self.z_codebook_level].get_codebook_entry(ix,shape=bhwc)
            #return shape is bchw
            #print('ix_quant.shape:',ix_quant.shape)
            ix_quant=einops.rearrange(ix_quant, 'b c h w  -> b (h w) c')
            # append to the sequence and continue
            #print('x.shape:',x.shape)
            #print('ix_quant.shape:',ix_quant.shape)
            x = torch.cat((x, ix_quant), dim=1)
        if self.transformer.cross_attention:
            # cut off sos token
            x = x[:,1:,:]
        else:
            # cut off conditioning
            x = x[:, c.shape[1]:,:]
        #print('return x_indices, shape:',x_indices.shape)
        return x_indices
