import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

# implement berhu loss function

def berhu_loss_function(prediction, target):
    #print(prediction.size())
    #print(target.size())
    #print(type(prediction)) # class 'torch.Tensor'
    #print(type(target)) # class 'torch.Tensor'
    # prediction, target = torch.from_numpy(prediction), torch.from_numpy(target)
    prediction, target = prediction.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)
    #prediction = torch.unsqueeze(prediction, 1)

    abs_error = torch.abs(prediction - target)

    # -----------------------------------------------------------------
    # Structure: torch.max(dim=None, keepdim=False)
    c,_ = torch.max(abs_error, dim=1, keepdim=True)
    c, _ = torch.max(c, dim=2, keepdim=True)
    c = c / (5.0)
    print('c shape', c.size())
    berhu_loss = torch.where(abs_error <= c, abs_error, (c**2 + abs_error**2)/(2*c))
    return berhu_loss

def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        n_classes = 20,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )
    # -----------------------------------------------------------------
    # Setup Metrics (substract one class)
    running_metrics_val = runningScore(n_classes-1)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()

    # get loss_seg meter and also loss_dep meter

    loss_seg_meter = averageMeter()
    loss_dep_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True



    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels, masks, depths) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            depths = depths.to(device)

            #print(images.shape)
            optimizer.zero_grad()
            outputs = model(images)
            #print('depths size: ', depths.size())
            #print('output shape: ', outputs.shape)


            loss_seg = loss_fn(input=outputs[:,:-1,:,:], target=labels)

            # -----------------------------------------------------------------
            # add depth loss

            # -----------------------------------------------------------------
            # MSE loss
            # loss_dep = F.mse_loss(input=outputs[:, -1,:,:], target=depths, reduction='mean')

            # -----------------------------------------------------------------
            # Berhu loss
            loss_dep = berhu_loss_function(prediction = outputs[:,-1,:,:], target=depths)
            #loss_dep = loss_dep.type(torch.cuda.ByteTensor)
            masks = masks.type(torch.cuda.ByteTensor)
            loss_dep = torch.sum(loss_dep[masks])/torch.sum(masks)
            print('loss depth', loss_dep)
            loss = loss_dep + loss_seg
            # -----------------------------------------------------------------

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  loss_seg: {:.4f}  loss_dep: {:.4f}  overall loss: {:.4f}   Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_seg.item(),
                    loss_dep.item(),
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"]:

                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, masks_val, depths_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        print('images_val shape', images_val.size())
                        # add depth to device
                        depths_val = depths_val.to(device)

                        outputs = model(images_val)
                        #depths_val = depths_val.data.resize_(depths_val.size(0), outputs.size(2), outputs.size(3))

                        # -----------------------------------------------------------------
                        # loss function for segmentation
                        print('output shape', outputs.size())
                        val_loss_seg = loss_fn(input=outputs[:,:-1,:,:], target=labels_val)

                        # -----------------------------------------------------------------
                        # MSE loss
                        # val_loss_dep = F.mse_loss(input=outputs[:, -1, :, :], target=depths_val, reduction='mean')


                        # -----------------------------------------------------------------
                        # berhu loss function
                        val_loss_dep = berhu_loss_function(prediction=outputs[:,-1, :, :], target=depths_val)
                        val_loss_dep = val_loss_dep.type(torch.cuda.ByteTensor)
                        masks_val = masks_val.type(torch.cuda.ByteTensor)
                        val_loss_dep = torch.sum(val_loss_dep[masks_val]) / torch.sum(masks_val)
                        val_loss = loss_dep + loss_seg
                        # -----------------------------------------------------------------

                        prediction = outputs[:,:-1, :, :]
                        prediction = prediction.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        # adapt metrics to seg and dep
                        running_metrics_val.update(gt, prediction)
                        loss_seg_meter.update(val_loss_seg.item())
                        loss_dep_meter.update(val_loss_dep.item())

                        # -----------------------------------------------------------------
                        # get rid of val_loss_meter
                        # val_loss_meter.update(val_loss.item())
                        # writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                        # logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
                        # -----------------------------------------------------------------


                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)


                print("Segmentation loss is {}".format(loss_seg_meter.avg))
                logger.info("Segmentation loss is {}".format(loss_seg_meter.avg))
                #writer.add_scalar("Segmentation loss is {}".format(loss_seg_meter.avg), i + 1)


                print("Depth loss is {}".format(loss_dep_meter.avg))
                logger.info("Depth loss is {}".format(loss_dep_meter.avg))
                #writer.add_scalar("Depth loss is {}".format(loss_dep_meter.avg), i + 1)


                val_loss_meter.reset()
                loss_seg_meter.reset()
                loss_dep_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

                    # insert print function to see if the losses are correct


            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_cityscapes.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)