import getopt
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, "../")

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(name)s  %(levelname)s %(message)s")


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(
            argv,
            "g:b:d:m:u:e:s:v:a:",
            [
                "gpu=",
                "batch_size=",
                "dataset=",
                "mode=",
                "univar=",
                "epochs=",
                "save_steps=",
                "val_steps=",
                "accumulation_steps=",
                "gradient_accumulation_steps=",
                "global_shuffle=",
                "shuffle_seed=",
                "shuffle_buffer_size=",
                "learning_rate=",
                "learning-rate=",
                "weight_decay=",
                "weight-decay=",
                "grad_clip_norm=",
                "grad-clip-norm=",
                "warmup_steps=",
                "warmup-steps=",
                "warmup_ratio=",
                "warmup-ratio=",
                "min_lr_ratio=",
                "min-lr-ratio=",
                "use_lr_schedule=",
                "use-lr-schedule=",
                "lr_scheduler_type=",
                "lr-scheduler-type=",
                "enable_revin=",
                "enable-revin=",
                "affine=",
                "point_loss=",
                "point-loss=",
                "huber_beta=",
                "huber-beta=",
            ],
        )
    except Exception:
        print("input error!")
        sys.exit(2)

    save_steps = 0
    val_steps = None
    gpu = None
    batch_size = None
    epochs = 30
    dataset = None
    mode = None
    univar = False
    gradient_accumulation_steps = 1
    global_shuffle = True
    shuffle_seed = 0
    shuffle_buffer_size = 0
    learning_rate = 1e-4
    weight_decay = 0.0
    grad_clip_norm = 1.0
    warmup_steps = 300
    warmup_ratio = 0.0
    min_lr_ratio = 0.0
    use_lr_schedule = True
    lr_scheduler_type = "linear"
    enable_revin = False
    affine = False
    point_loss = "huber"
    huber_beta = 1.0

    for opt, arg in opts:
        if opt in ["-g", "--gpu"]:
            gpu = arg
        elif opt in ["-b", "--batch_size"]:
            batch_size = int(arg)
        elif opt in ["-d", "--dataset"]:
            dataset = arg
        elif opt in ["-m", "--mode"]:
            mode = arg
        elif opt in ["-u", "--univar"]:
            univar = bool(int(arg))
        elif opt in ["-e", "--epochs"]:
            epochs = int(arg)
        elif opt in ["-s", "--save_steps"]:
            save_steps = int(arg)
        elif opt in ["-v", "--val_steps"]:
            val_steps = int(arg)
        elif opt in ["-a", "--accumulation_steps", "--gradient_accumulation_steps"]:
            gradient_accumulation_steps = int(arg)
        elif opt == "--global_shuffle":
            global_shuffle = bool(int(arg))
        elif opt == "--shuffle_seed":
            shuffle_seed = int(arg)
        elif opt == "--shuffle_buffer_size":
            shuffle_buffer_size = int(arg)
        elif opt in ["--learning_rate", "--learning-rate"]:
            learning_rate = float(arg)
        elif opt in ["--weight_decay", "--weight-decay"]:
            weight_decay = float(arg)
        elif opt in ["--grad_clip_norm", "--grad-clip-norm"]:
            grad_clip_norm = float(arg)
        elif opt in ["--warmup_steps", "--warmup-steps"]:
            warmup_steps = int(arg)
        elif opt in ["--warmup_ratio", "--warmup-ratio"]:
            warmup_ratio = float(arg)
        elif opt in ["--min_lr_ratio", "--min-lr-ratio"]:
            min_lr_ratio = float(arg)
        elif opt in ["--use_lr_schedule", "--use-lr-schedule"]:
            use_lr_schedule = bool(int(arg))
        elif opt in ["--lr_scheduler_type", "--lr-scheduler-type"]:
            lr_scheduler_type = str(arg)
        elif opt in ["--enable_revin", "--enable-revin"]:
            enable_revin = bool(int(arg))
        elif opt == "--affine":
            affine = bool(int(arg))
        elif opt in ["--point_loss", "--point-loss"]:
            point_loss = str(arg).lower()
        elif opt in ["--huber_beta", "--huber-beta"]:
            huber_beta = float(arg)
        else:
            print("input error!")
            sys.exit(2)

    if gpu is None or batch_size is None:
        print("input error!")
        sys.exit(2)
    if point_loss not in {"mse", "huber"}:
        print("input error: --point_loss must be mse or huber")
        sys.exit(2)
    if huber_beta <= 0:
        print("input error: --huber_beta must be > 0")
        sys.exit(2)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    from utils import dataset_loader
    from core.model import GTT, ModelConfig

    input_len = 1984
    pred_len = 64
    train_list, val_list, test_list = dataset_loader.load_data(dataset, prediction_length=pred_len)
    pm = None
    project_dir = Path(__file__).resolve().parent
    cp = str((project_dir / "../../autodl-tmp/GTT-pytorch" / f"GTT-finetune-{mode}").resolve())
    mc = ModelConfig(
        block_size=1984,
        patch_size=64,
        pred_len=pred_len,
        enable_revin=enable_revin,
        affine=affine,
        revin_time=True,
        n_embd=768,
        encoder_layers=8,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        embedd_pdrop=0.0,
        dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        encoder_layerdrop=0.0,
        point_loss=point_loss,
        huber_beta=huber_beta,
    )
    model = GTT(configs=mc)
    hist = model.train(
        train_list,
        val_list,
        cp,
        pm=pm,
        batch_size=batch_size,
        epochs=epochs,
        distribute=True,
        verbose=1,
        save_steps=save_steps,
        val_steps=val_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        use_lr_schedule=use_lr_schedule,
        lr_scheduler_type=lr_scheduler_type,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_shuffle=global_shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    model.save_model(
        cp,
        hist=hist,
        training_config={
            "dataset": dataset,
            "mode": mode,
            "gpu": gpu,
            "batch_size": batch_size,
            "epochs": epochs,
            "save_steps": save_steps,
            "val_steps": val_steps,
            "input_len": input_len,
            "pred_len": pred_len,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "warmup_steps": warmup_steps,
            "warmup_ratio": warmup_ratio,
            "min_lr_ratio": min_lr_ratio,
            "use_lr_schedule": use_lr_schedule,
            "lr_scheduler_type": lr_scheduler_type,
            "enable_revin": enable_revin,
            "affine": affine,
            "point_loss": point_loss,
            "huber_beta": huber_beta,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_shuffle": global_shuffle,
            "shuffle_seed": shuffle_seed,
            "shuffle_buffer_size": shuffle_buffer_size,
            "checkpoint_dir": cp,
        },
    )

# train with dataset A
# python train.py \
#   -g 0,1,2,3,4,5,6,7 \
#   -b 64 \
#   -d "../dataset/A" \
#   -m A_eval \
#   -e 1 \
#   -s 0

# train with datasets A+B
# python train.py \
#   -g 0,1,2,3,4,5,6,7 \
#   -b 64 \
#   -d "../dataset/A+../dataset/B" \
#   -m A_eval \
#   -e 1 \
#   -s 0
