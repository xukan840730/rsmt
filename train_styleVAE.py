#import argparse
import copy
import os
import re
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything

from src.Datasets.BaseLoader import WindowBasedLoader
from src.Net.StyleVAENet import StyleVAENet
from src.utils import BVH_mod as BVH


def setup_seed(seed:int):
    seed_everything(seed,True)
def test_model():
    dict = {}
    #dict['fast_dev_run'] = 1 # only run 1 train, val, test batch and program ends
    dict['limit_train_batches'] = 0.1
    dict['limit_val_batches'] = 0.7
    return dict
def detect_nan_par():
    '''track_grad_norm": 'inf'''
    return { "detect_anomaly":True}
def select_gpu_par():
    return {"accelerator":'gpu', "auto_select_gpus":True, "devices":-1}

def create_common_states(prefix:str):
    log_name = prefix+'/'
    '''test upload'''
    parser = ArgumentParser()
    parser.add_argument("--dev_run", action="store_true")
    parser.add_argument("--version", type=str, default="-1")
    parser.add_argument("--epoch",type=str,default="last")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    ckpt_path_prefix = "results/"
    if (args.version != "-1"):
        version = args.version
    else:
        version = None
    '''Create Loggers tensorboard'''
    if args.dev_run:
        log_name += "dev_run"
    else:
        log_name += "myResults"
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="tensorboard_logs/", name=log_name, version=None)
    load_ckpt_path = os.path.join(ckpt_path_prefix, prefix+'/myResults', str(version))
    save_ckpt_path = os.path.join(ckpt_path_prefix, log_name, str(tb_logger.version))

    if (args.resume == True):
        check_file = load_ckpt_path+"/"
        if (args.epoch == "last"):
            check_file += "last.ckpt"
        else:
            dirs = os.listdir(check_file)
            for dir in dirs:
                st = "epoch=" + args.epoch + "-step=\d+.ckpt"
                out = re.findall(st, dir)
                if (len(out) > 0):
                    check_file += out[0]
                    print(check_file)
                    break
        resume_from_checkpoint = check_file  # results/version/last.ckpt"
    else:
        resume_from_checkpoint = None
    checkpoint_callback = [ModelCheckpoint(dirpath=save_ckpt_path + "/", save_top_k=-1, save_last=True, every_n_epochs=5),
                           ModelCheckpoint(dirpath=save_ckpt_path + "/", save_top_k=1, monitor="val_loss", save_last=False, every_n_epochs=1,save_weights_only=True),
                          # EMA(0.99)
                           ]
    '''Train'''
    checkpoint_callback[0].CHECKPOINT_NAME_LAST = "last"
    profiler = SimpleProfiler()#PyTorchProfiler(filename="profiler")
    trainer_dict = {
        "callbacks":checkpoint_callback,
        "profiler":profiler,
        "logger":tb_logger
    }
    return args,trainer_dict,resume_from_checkpoint,load_ckpt_path


# khanxu: export torch model to onnx
def export_embedding_encoder_to_onnx(model):
    # assert isinstance(model, EmbeddingTwoFrameEncoder)
    model.cpu()  # change model to cpu before exporting
    model.eval()

    dummy_input = torch.randn((1, 444), requires_grad=True)

    # export the model:
    torch.onnx.export(model,
                      dummy_input,
                      "embedding_two_frame_encoder.onnx",
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input_condition'],
                      output_names=['output_latent', 'output_mu', 'output_logvar'],
                      dynamic_axes={'input_condition': {0: 'batch_size'},
                                    'output_latent': {0: 'batch_size'},
                                    'output_mu': {0: 'batch_size'},
                                    'output_logvar': {0: 'batch_size'}})
    print('export embedding_encoder to onnx done!')


def export_moe_gate_decoder_to_onnx(model):
    # assert isinstance(model, MoeGateDecoder)
    model.cpu()  # change model to cpu before exporting
    model.eval()

    input1 = torch.randn((1, 32), requires_grad=True)
    input2 = torch.randn((1, 231), requires_grad=True)
    input3 = torch.randn((1, 10, 2), requires_grad=True)

    # export the model:
    torch.onnx.export(model,
                      args=(input1, input2, input3),
                      f='moe_gate_decoder.onnx',
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input_latent', 'input_condition', 'input_phase'],
                      output_names=['output_pred_pose', 'output_coeffs'],
                      dynamic_axes={'input_latent': {0: 'batch_size'},
                                    'input_condition': {0: 'batch_size'},
                                    'input_phase': {0: 'batch_size'},
                                    'output_pred_pose': {0: 'batch_size'},
                                    'output_coeffs': {0: 'batch_size'}, }
                      )
    print('export moe_gate_decoder to onnx done!')


def training_style100():
    from src.Datasets.StyleVAE_DataModule import StyleVAE_DataModule
    from src.Datasets.Style100Processor import StyleLoader
    from src.Net.StyleVAENet import Application_StyleVAE,VAEMode
    prefix = "StyleVAE2"
    data_set = "style100"
    prefix += "_" + data_set
    args, trainer_dict, resume_from_checkpoint, ckpt_path = create_common_states(prefix)
    resume_from_checkpoint = None
    loader = WindowBasedLoader(61, 21, 1)
    dt = 1. / 30.
    phase_dim = 10
    phase_file = "+phase_gv10"
    latent_size = 32
    net_mode = VAEMode.SINGLE

    batch_size = 32
    if (args.test == False):
        '''Create the model'''
        style_loader = StyleLoader()
        data_module = StyleVAE_DataModule(style_loader, phase_file + loader.get_postfix_str(),style_file_name=None, dt=dt, batch_size=batch_size, mirror=0.0)  # when apply phase, should avoid mirror
        model = StyleVAENet(data_module.skeleton,  phase_dim=phase_dim, latent_size=latent_size,batch_size=batch_size,mode='pretrain',net_mode=net_mode)
        if (args.dev_run):
            trainer = Trainer(**trainer_dict, **test_model(),
                              **select_gpu_par(), precision=32, reload_dataloaders_every_n_epochs=1,#gradient_clip_val=1.0,#**detect_nan_par(),
                              log_every_n_steps=5, flush_logs_every_n_steps=10,
                              weights_summary='full')
        else:

            trainer = Trainer(**trainer_dict, max_epochs=10000, reload_dataloaders_every_n_epochs=1,gradient_clip_val=1.0,#**detect_nan_par(),
                              **select_gpu_par(), log_every_n_steps=50,
                              flush_logs_every_n_steps=100)
        trainer.fit(model, datamodule=data_module)

    else:

        style_loader = StyleLoader()
        data_module = StyleVAE_DataModule(style_loader, phase_file + loader.get_postfix_str(),None, dt=dt, batch_size=batch_size, mirror=0.0)
        data_module.setup()
        check_file = ckpt_path + "/"
        if (args.epoch == "last"):
            check_file += "last.ckpt"
            print(check_file)
        else:
            dirs = os.listdir(check_file)
            for dir in dirs:
                st = "epoch=" + args.epoch + "-step=\d+.ckpt"
                out = re.findall(st, dir)
                if (len(out) > 0):
                    check_file += out[0]
                    print(check_file)
                    break
        model = StyleVAENet.load_from_checkpoint(check_file, moe_decoder=None,pose_channels=6,net_mode=net_mode,strict=False)
        model = model.cuda()
        src_motion = data_module.test_set.dataset["HighKnees"][0]
        print(len(data_module.test_set.dataset["HighKnees"]))
        source = BVH.read_bvh("source_template.bvh")
        '''check if space can produce netural space: encoding=False, style=kick'''
        data_module.mirror = 0
        model = model.cpu()
        model.eval()
        app = Application_StyleVAE(model, data_module)
        app = app.float()
        app.setSource(src_motion)
        output = copy.deepcopy(source)
        output.hip_pos, output.quats = app.forward_and_plot(seed=3000, encoding=True)
        BVH.save_bvh(f"test_net__styleVAE_output__version_{args.version}.bvh", output)
        source.hip_pos, source.quats = app.get_source()
        BVH.save_bvh(f"source__styleVAE_output__version_{args.version}.bvh", source)
        torch.save(model, ckpt_path + "/m_save_model_" + str(args.epoch))

        # try export models to onnx
        export_embedding_encoder_to_onnx(model.embedding_encoder)
        export_moe_gate_decoder_to_onnx(model.decoder)


if __name__ == '__main__':
    setup_seed(3407)
    training_style100()


