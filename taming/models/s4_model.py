import os, math
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import einops
from matplotlib import pyplot as plt

from main import instantiate_from_config
from taming.modules.util import SOSProvider
from taming.modules.state_spaces.s4d import S4DList


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetS4(pl.LightningModule):
    def __init__(self,
                 s4_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0.,
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
        self.s4_model = instantiate_from_config(config=s4_config)

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
        raise NotImplementedError

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
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
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
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        return NotImplementedError

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

    #Copied from Kim's repo, working version (the one he sent later)
    def configure_optimizers(self):
        weight_decay = 0.01
        patience = 10
        #opt = torch.optim.Adam(self.s4_model.parameters(), lr=lr, betas=(0.5, 0.9))
        #return opt

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.s4_model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                if hasattr(p, "_optim"):
                    no_decay.add(fpn)
                else:
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    else:
                        if pn[-1] == "C" or pn[-1] == "D":
                            decay.add(fpn)
                        #for mn2, m2 in m.named_modules():
                        #    print(mn2, m2)
                        #print("missing", fpn, m.__class__.__name__)
        # special case the position embedding parameter in the root GPT module as not decayed
        #no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.s4_model.named_parameters()}
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

    """
    def configure_optimizers(self):
       " 
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
       " 
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.s4_model.named_modules():
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
                else:
                    no_decay.add(fpn)

        #if self.joint_training:
        #    no_decay.add('level_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.s4_model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        decay = decay - inter_params
        inter_params = decay & no_decay
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
    """

class RVQS4 (Net2NetS4):
    def __init__(self,
                 s4_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0.,
                 unconditional=False,
                 z_codebook_level=0,
                 joint_training=False,
                 end_to_end_sampling=False,
                 input_1d=False
                 ):

        #Adjust vocab size of transformer to that of the chosen codebook level
        self.vocab_size=first_stage_config.params.n_embeds[z_codebook_level]
        s4_config.params.codebook_size=self.vocab_size

        #s4_config.params.device=self.device
        s4_config.params.inp_dim = first_stage_config.params.embed_dim

        #if joint_training:
        #    s4_config.params.n_codebook_levels = first_stage_config.params.n_levels

        self.first_stage_config=first_stage_config
        self.s4_config=s4_config

        self.joint_training = joint_training
        self.end_to_end_sampling = end_to_end_sampling
        self.input_1d = input_1d

        if input_1d:
            s4_config.params.inp_dim = 1

        if joint_training:
            #self.z_codebook_level is altered in each train loop in joint_training mode, starts with 0
            z_codebook_level = 0
        self.z_codebook_level = z_codebook_level
        self.c_codebook_level = z_codebook_level-1 if z_codebook_level>0 else None


        print('Constructing RVQTransformer object. s4_config: ', s4_config)
        super().__init__(
                 s4_config,
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

        self.sos_token=sos_token

        #self.sos_emb = nn.Parameter(torch.rand(1,1,s4_config.params.inp_dim))
        self.uncond_emb = nn.Parameter(torch.zeros(1,1,s4_config.params.inp_dim))

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

    #Shape of cb_index is (bs,1), output shape is bchw
    def get_codebook_entry_one_batch(self,cb_index,cb_level):
        cb_dim = self.first_stage_config.params.embed_dim
        bhwc=(cb_index.shape[0],1,1,cb_dim)
        return self.first_stage_model.quantizers[cb_level].get_codebook_entry(cb_index,shape=bhwc)

    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
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


    @torch.no_grad()
    def encode_to_z(self, x, codebook_level=None):
        #print('------------------------encode_to_z called-------------------------------')
        #print('x.shape:',x.shape)
        if codebook_level is None:
            codebook_level=self.z_codebook_level

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

        if self.be_unconditional:
            hidden_dim=self.get_hidden_dim()
            bs=c.shape[0]
            bchw=(bs,self.first_stage_config.params.embed_dim,hidden_dim,hidden_dim)

            #Condition on image made up of trainable 'no input embedding' vectors for unconditional case
            quant_c = self.uncond_emb.expand(bs,hidden_dim**2,-1)
            #quant_c = torch.ones(*bchw,device=c.device) * self.sos_token
            #quant_c = einops.rearrange(quant_c, 'b c h w -> b (h w) c')
            indices = None

        else:
            pre_quant = self.cond_stage_model.encode_to_prequant(c)
            all_quantized, all_indices, all_losses = self.cond_stage_model.make_quantizations(pre_quant)

            #The quantizations in all_quantized are already cumulative over the codebook levels
            quant_c = all_quantized[codebook_level]

            #print('quant_c.shape:',quant_c.shape)
            #print('all_indices len: ',len(all_indices))
            indices=all_indices[codebook_level]
            indices = einops.rearrange(indices, '(b hw) -> b hw', b=quant_c.shape[0]) 

            bchw = quant_c.shape
            quant_c = einops.rearrange(quant_c, 'b c h w -> b (h w) c')

        return quant_c, indices, bchw

    def init_cond_stage_from_ckpt(self, config):
        if self.z_codebook_level==0:
            print('First stage has lowest codebook level. Using a dummy array of zeros as conditioning image.')
            self.be_unconditional = True
        else:
            print('First stage codebook level is {}. Using codebook_level {} as cond stage'.format(self.z_codebook_level,self.c_codebook_level))
            self.be_unconditional = False

        self.cond_stage_model = self.first_stage_model
        self.cond_stage_key = self.first_stage_key

            
    # one step to produce the logits
    def forward(self, x, c):
        #print('forward, x.shape:',x.shape)
        #print('forward, c.shape:',c.shape)
        z_quant, z_indices, _ = self.encode_to_z(x)

        debug=True
        if debug:
            c_quant = z_quant[:,:0,:]
            c_indices = z_quant[:,:0]
        else:
            c_quant, c_indices, _ = self.encode_to_c(c)



        if c_quant.shape[1]==0:
            sos_quant = self.get_codebook_entry_one_batch(torch.ones_like(z_indices)[:,0]*self.sos_token,cb_level=self.z_codebook_level)
            sos_quant = einops.rearrange(sos_quant, 'b c h w -> b (h w) c')

            cz_quants = torch.cat((sos_quant, z_quant), dim=1)
        else:
            cz_quants = torch.cat((c_quant, z_quant), dim=1)

        target = z_indices
        #print('target.shape: ',target.shape)
        #print('cz_quants.shape:', cz_quants.shape)

        #logits, _ = self.s4_model(cz_quants[:,:-1,:])
        logits, _ = self.s4_model(cz_quants)
        
        #print('logits.shape: ',logits.shape)

        # cut off conditioning outputs
        #TODO: Think about if conditioned case needs SOS token as well
        #TODO: Check how S4 is handling the logits/input shift in CNN mode, is first logit corresponding to prediction of first token or of second token?
        #Here: Suppose that the shift is already taken care of, so first logit corresponds to SOS token
        if c_quant.shape[1]>0:
            logits = logits[:, c_quant.shape[1]:]
        else:
            logits = logits[:, 1:]

        #print('logits.shape: ',logits.shape)


        return logits, target

    #TODO: Loss Function for CodeGPT version with vector input!
    def shared_step(self, batch, batch_idx):
        #print('=====================shared_step===============')

        if self.joint_training:
            self.z_codebook_level = 0
            self.c_codebook_level = None
            self.be_unconditional = True

            x, c = self.get_xc(batch)
            logits, target = self(x, c)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            self.be_unconditional = False

            for z_lvl in range(1,self.first_stage_model.n_levels):
                #print('==========loop, z_lvl=', z_lvl)
                self.z_codebook_level = z_lvl
                self.c_codebook_level = z_lvl-1

                x, c = self.get_xc(batch)
                logits, target = self(x, c)
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
    def log_images_one_lvl(self, batch, return_quantized_sample=False, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):


        sample_debug=True

        #print('=====log_images called=====')
        log = dict()

        N = kwargs['N'] if 'N' in kwargs.keys() else 4

        x, c = self.get_xc(batch, N)

        x = x.to(device=self.device)
        c = c.to(device=self.device)

        #quant_z has shape b (h w) c
        quant_z, z_indices, z_bchw = self.encode_to_z(x)
        quant_c, c_indices, c_bchw = self.encode_to_c(c)


        log_prediction=True

        if log_prediction and self.z_codebook_level==0:
            logits, _ = self.forward(x,c) # (B,L,C)
            logits = einops.rearrange(logits, 'b l c -> (b l) c')
            # Output distribution
            probs = F.softmax(logits, dim=-1)

            # Optional: scale by temperature
            if temperature is not None:
                logits = logits / temperature
            
            # Optional: top_k sampling
            if top_k is not None:
                v, ix = torch.topk(probs, top_k)
                logits[logits < v[..., [-1]]] = -1e20

            # Sample from the distribution
            logits = torch.distributions.Categorical(logits=logits).sample()
            
            predictions = self.decode_to_img(logits, z_bchw)
            log['predictions'] = predictions


        #TODO: Adapt half picture sampling to work with CodeGPT
        cond_perc_list=[0.2,0.5,0.8]
        cond_perc_outputs=[]
        for cond_on_first_x_percent in cond_perc_list:
            sos_index=torch.ones_like(z_indices)[:,:1]*self.sos_token
            z_start_indices = z_indices[:,:round(quant_z.shape[1]*cond_on_first_x_percent)]
            z_start_indices = torch.cat([sos_index,z_start_indices],dim=1)

            n_tokens_to_sample = z_indices.shape[1] + 1 - z_start_indices.shape[1]

            index_sample = self.sample_s4(z_start_indices, quant_c,
                                       steps=n_tokens_to_sample,
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else min(100,self.s4_config.params.codebook_size),
                                       callback=callback if callback is not None else lambda k: None)

            #cut off SOS token
            index_sample = index_sample[:,1:]
            #print('Sampling with start indices, index_sample.shape:',index_sample.shape)

            if self.be_unconditional:
                if sample_debug:
                    cond_perc_outputs.append(self.decode_to_img(index_sample, z_bchw))
                else:
                    #dummy_c = torch.zeros(z_bchw[0],z_bchw[2]*z_bchw[3],z_bchw[1],device=quant_c.device,dtype=quant_c.dtype)
                    dummy_c = self.uncond_emb.expand(z_bchw[0],z_bchw[2]*z_bchw[3],-1)
                    x_sample_nopix_sum = self.cond_and_pred_to_img(dummy_c, index_sample, z_bchw)
            else:
                x_sample_nopix_sum = self.cond_and_pred_to_img(quant_c, index_sample, z_bchw)



        #print('------------Sampling-------------')
        #print('quant_z.shape:,', quant_z.shape)
        #print('quant_c.shape:,', quant_c.shape)

        #print('===========normal sampling==========\n')
        #z_start_quants=quant_z[:, :0, :]
        sos_index=torch.ones_like(z_indices)[:,:1]*self.sos_token
        z_start_indices = sos_index
        n_tokens_to_sample = z_indices.shape[1]

        #print('z_start_quants.shape:',z_start_quants.shape)
        index_sample = self.sample_s4(z_start_indices, quant_c,
                                   steps=n_tokens_to_sample,
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else min(100,self.s4_config.params.codebook_size),
                                   callback=callback if callback is not None else lambda k: None)


        #print('Normal sampling, index_sample.shape:',index_sample.shape)

        #cut off SOS token
        index_sample = index_sample[:,1:]

        #plt.hist(index_sample.cpu(),bins=2048)
        #plt.savefig("/export/home/fmayer/taming-transformers/logs/cb_histograms/",dpi=2000)

        if self.be_unconditional:
            if sample_debug:
                x_sample_nopix_sum = self.decode_to_img(index_sample, z_bchw)
            else:
                #dummy_c = torch.zeros(z_bchw[0],z_bchw[2]*z_bchw[3],z_bchw[1],device=quant_c.device,dtype=quant_c.dtype)
                dummy_c = self.uncond_emb.expand(z_bchw[0],z_bchw[2]*z_bchw[3],-1)
                x_sample_nopix_sum = self.cond_and_pred_to_img(dummy_c, index_sample, z_bchw)
        else:
            x_sample_nopix_sum = self.cond_and_pred_to_img(quant_c, index_sample, z_bchw)


        #Det sample - deactivated
        create_det_sample=False
        if create_det_sample:
            z_start_quants=quant_z[:, :0, :]
            index_sample_det = self.sample_s4(z_start_quants, quant_c,
                                       steps=z_indices.shape[1],
                                       sample=False,
                                       callback=callback if callback is not None else lambda k: None)

            if self.be_unconditional:
                dummy_c = torch.zeros(z_bchw[0],z_bchw[2]*z_bchw[3],z_bchw[1],device=quant_c.device,dtype=quant_c.dtype)
                x_sample_det = self.cond_and_pred_to_img(dummy_c, index_sample_det, z_bchw)
            else:
                x_sample_det = self.cond_and_pred_to_img(quant_c, index_sample_det, z_bchw)


        #log input images, as well as conditioning and target image (both ResVQGAN reconstructions at different codebook levels)
        log_resvq = self.first_stage_model.log_images(batch)
        log["inputs"] = x
        log["reconstructions_target_lvl_{}".format(self.z_codebook_level)] = log_resvq["reconstructions_{}".format(self.z_codebook_level)]

        if not self.joint_training and self.z_codebook_level>0:
            log["reconstructions_cond_lvl_{}".format(self.c_codebook_level)] = log_resvq["reconstructions_{}".format(self.c_codebook_level)]

        if self.z_codebook_level==0:
            log["sample_end2end_lvl{}".format(self.z_codebook_level)] = x_sample_nopix_sum
        else:
            log["samples_nopix_lvl{}".format(self.z_codebook_level)] = x_sample_nopix_sum
            #log["samples_det_sum_cond"] = x_sample_det_sum

        for i in range(len(cond_perc_list)):
            log["sample_{}_perc_given_lvl{}".format(round(cond_perc_list[i]*100),self.z_codebook_level)] = cond_perc_outputs[i]

        if return_quantized_sample:
            quant_sample = self.quant_c_and_ind_to_next_cblvl(quant_c, index_sample, z_bchw)
            return log, quant_sample
        else:
            return log


    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, log_ground_truth_upsampling=True, **kwargs):
        print('==========Starting Image Logging=========')

        if not self.joint_training:
            logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface, **kwargs)
            return logs

        else:
            self.z_codebook_level = 0
            self.c_codebook_level = None
            self.be_unconditional = True

            print('==========Logging level 0')

            if not self.end_to_end_sampling:
                logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)

            else:
                logs, quant_sample = self.log_images_one_lvl(batch, return_quantized_sample=True, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)
                logs['sample_end2end_lvl0'] = logs['samples_nopix_lvl0']

            self.be_unconditional = False

            for z_lvl in range(1,self.first_stage_model.n_levels):
                print('==========loop, z_lvl=', z_lvl)
                self.z_codebook_level = z_lvl
                self.c_codebook_level = z_lvl-1

                #Sample residuals to get from ground truth lvl n to lvl n+1 (errors of previous levels are not propagated for this case)
                if log_ground_truth_upsampling:
                    new_logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)
                else:
                    new_logs={}

                #Sample residuals to get from sampled lvl n to lvl n+1
                if self.end_to_end_sampling:
                    z_bchw = quant_sample.shape
                    quant_sample = einops.rearrange(quant_sample,'b c h w -> b (h w) c')
                    z_start_quants = quant_sample[:, :0, :]

                    #Sample new residual indices, using last quantized sample as conditioning
                    new_sample_ind = self.sample_s4(z_start_quants, c=quant_sample,
                                               steps=quant_sample.shape[1],
                                               sample=True)


                    #Add up predictions and last level quants to new quantized sample
                    quant_sample = self.quant_c_and_ind_to_next_cblvl(quant_sample, new_sample_ind, target_shape_bchw=z_bchw)
                    x_sample_end2end = self.first_stage_model.decode(quant_sample)

                    new_logs['sample_end2end_lvl{}'.format(self.z_codebook_level)]=x_sample_end2end


                logs = {**logs, **new_logs}

            return logs

    def log_images_one_sequence(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, log_ground_truth_upsampling=True, **kwargs):
        print('==========Starting Image Logging, all levels in one sequence=========')

        if not self.joint_training:
            logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface, **kwargs)
            return logs

        else:
            self.z_codebook_level = 0
            self.c_codebook_level = None
            self.be_unconditional = True

            print('==========Logging level 0')

            if not self.end_to_end_sampling:
                logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)

            else:
                logs, quant_sample = self.log_images_one_lvl(batch, return_quantized_sample=True, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)
                logs['sample_end2end_lvl0'] = logs['samples_nopix_lvl0']

            self.be_unconditional = False

            for z_lvl in range(1,self.first_stage_model.n_levels):
                print('==========loop, z_lvl=', z_lvl)
                self.z_codebook_level = z_lvl
                self.c_codebook_level = z_lvl-1

                #Sample residuals to get from ground truth lvl n to lvl n+1 (errors of previous levels are not propagated for this case)
                if log_ground_truth_upsampling:
                    new_logs = self.log_images_one_lvl(batch, temperature=temperature, top_k=top_k, callback=callback, lr_interface=lr_interface,**kwargs)
                else:
                    new_logs={}

                #Sample residuals to get from sampled lvl n to lvl n+1
                if self.end_to_end_sampling:
                    z_bchw = quant_sample.shape
                    quant_sample = einops.rearrange(quant_sample,'b c h w -> b (h w) c')
                    z_start_quants = quant_sample[:, :0, :]

                    #Sample new residual indices, using last quantized sample as conditioning
                    new_sample_ind = self.sample_s4(z_start_quants, c=quant_sample,
                                               steps=quant_sample.shape[1],
                                               sample=True)


                    #Add up predictions and last level quants to new quantized sample
                    quant_sample = self.quant_c_and_ind_to_next_cblvl(quant_sample, new_sample_ind, target_shape_bchw=z_bchw)
                    x_sample_end2end = self.first_stage_model.decode(quant_sample)

                    new_logs['sample_end2end_lvl{}'.format(self.z_codebook_level)]=x_sample_end2end


                logs = {**logs, **new_logs}

            return logs

    def sample_s4(self, x_start_indices, c, steps, temperature=1., sample=True, top_k=None,
               callback=lambda k: None):

        #temperature=100.
        #top_k=1000
        #print('sample_s4========================================')
        #print('steps:',steps)
        #print('c.shape:',c.shape)

        #print('x_start_indices.shape:',x_start_indices.shape)
        assert x_start_indices.shape[1]>0
        assert torch.isnan(c).sum()==0
        assert torch.isnan(x_start_indices).sum()==0

        #x = torch.cat((c,x),dim=1)

        #state shape: (bs,tok_emb_dim,state_dim)
        #state = torch.rand((x.shape[0],self.s4_config.params.tok_emb_dim,self.s4_config.params.state_dim),device=x.device, dtype=x.dtype)

        state = self.s4_model.default_state(x_start_indices.shape[0])

        #x_indices = torch.zeros((x.shape[0],0),device=x.device,dtype=torch.long)

        self.s4_model.setup_step()


        #Let the model see all the start indices to get to the right state, the logit outputs are not used
        for k in range(x_start_indices.shape[1]):
            ix = x_start_indices[:,k]

            #bhwc=(z_bchw[0],z_bchw[2],z_bchw[3],z_bchw[1])
            ix_quant = self.get_codebook_entry_one_batch(cb_index=ix,cb_level=self.z_codebook_level)
            #ix_quant = self.first_stage_model.quantizers[self.z_codebook_level].get_codebook_entry(ix,shape=bhwc)
            ix_quant=einops.rearrange(ix_quant, 'b c h w  -> b (h w) c')

            x = ix_quant[:,0,:]

            logits, state = self.s4_model.step(x, state)


        #Starting from the state that contains the start indices, sample the rest of the image
        x_indices = x_start_indices
        for k in range(steps):
            assert torch.isnan(x).sum()==0
            callback(k)

            #For the first step, use the last logit output of the previous loop
            if k>0:
                logits, state = self.s4_model.step(x, state)

            # pluck the logits at the final step and scale by temperature
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            if k==0:
                plt.figure(2)
                #plt.hist(probs.flatten().cpu(),bins=10)
                #plt.savefig("/export/home/fmayer/taming-transformers/logs/cb_histograms/probs.png")

            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            x_indices=torch.cat((x_indices,ix), dim=1)
            #bhwc=(ix.shape[0],1,1,x.shape[1])
            ix_quant = self.get_codebook_entry_one_batch(cb_index=ix,cb_level=self.z_codebook_level)
            #ix_quant = self.first_stage_model.quantizers[self.z_codebook_level].get_codebook_entry(ix,shape=bhwc)
            ix_quant=einops.rearrange(ix_quant, 'b c h w  -> b (h w) c')

            #x = torch.cat((x, ix_quant), dim=1)
            x = ix_quant[:,0,:]

        return x_indices



