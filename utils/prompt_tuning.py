import argparse
import os.path
import warnings
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from dassl.data.datasets import one_shot
import datasets.one_shot
import datasets.oxford_pets
import utils.coop


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.shots:
        cfg.DATASET.SHOTS = args.shots

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.nctx:
        cfg.TRAINER.COOP.N_CTX = args.nctx

    if args.csc:
        cfg.TRAINER.COOP.CSC = args.csc

    if args.position:
        cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = args.position

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, dataset_name):
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.CSC = True  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 10
    cfg.DATALOADER.TEST.BATCH_SIZE = 10
    cfg.DATALOADER.NUM_WORKERS = 1

    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bicubic"
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]

    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.002
    cfg.OPTIM.MAX_EPOCH = 50
    cfg.OPTIM.LR_SCHEDULER = "cosine"
    cfg.OPTIM.WARMUP_EPOCH = 1
    cfg.OPTIM.WARMUP_TYPE = "constant"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5

    cfg.TRAIN.PRINT_FREQ = 5

    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"

    cfg.OUTPUT_DIR = os.path.join('weight', dataset_name)
    cfg.DATASETNAME = dataset_name


def setup_cfg(args, dataset_name):
    cfg = get_cfg_default()
    extend_cfg(cfg, dataset_name)

    # 1. From the dataset config file
    if args.dataset:
        cfg.DATASET.NAME = args.dataset

    if args.position:
        cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = args.position

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="one_shot")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--shots", type=str, default="1")
    parser.add_argument("--nctx", type=int, default=8)
    parser.add_argument("--csc", type=str, default="True")
    parser.add_argument("--position", type=str, default="end")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )

    parser.add_argument("--trainer", type=str, default="CoOp", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    return args


def prompt_tuning(dataset_name):
    args = get_arguments()
    warnings.filterwarnings("ignore")

    cfg = setup_cfg(args, dataset_name)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    # setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()