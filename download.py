from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


if __name__ == '__main__':

    # Use this command for evaluate the GLPT-T model
    # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
    config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
    weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

    # Use this command to evaluate the GLPT-L model
    # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
    # config_file = "configs/pretrain/glip_Swin_L.yaml"
    # weight_file = "MODEL/glip_large_model.pth"

    # update the config options with the config file
    # manual override some options
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )