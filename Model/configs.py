import ml_collections

"""The Settting of VIT"""
def get_3DReg_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4, 4)})
    config.patches.grid = (4, 4, 4)
    config.hidden_size = 252
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 4

    config.conv_first_channel = 512
    config.down_num = 2
    config.n_dims = 3
    return config