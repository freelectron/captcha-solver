import torch
import torch.nn as nn

CNN_BASIC = [
    {"layer": "Conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 1}},
    {"layer": "ReLU", "params": {}},
    {"layer": "Conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 1, "padding": 1}},
    {"layer": "ReLU", "params": {}},
    # {"layer": "AdaptiveAvgPool2d", "params": {"output_size": 2}},
    {"layer": "Conv2d", "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
    {"layer": "ReLU", "params": {}},
]

RNN_PART = [
    {"layer": "LSTM", "params": {"input_size": 256, "hidden_size": 128}},
]

def parse_crrn_config(config):
    layers = []
    for layer_info in config:
        layer_type = layer_info["layer"]
        params = layer_info["params"]
        if layer_type == "Conv2d":
            layers.append(nn.Conv2d(**params))
        elif layer_type == "ReLU":
            layers.append(nn.ReLU())
        elif layer_type == "AdaptiveAvgPool2d":
            layers.append(nn.AdaptiveAvgPool2d(**params))
        elif layer_type == "Flatten":
            layers.append(nn.Flatten())
        elif layer_type == "Linear":
            layers.append(nn.Linear(**params))
        elif layer_type == "BatchNorm2d":
            layers.append(nn.BatchNorm2d(**params))
        elif layer_type == "Dropout":
            layers.append(nn.Dropout(**params))
        elif layer_type == "LSTM":
            layers.append(nn.LSTM(**params))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return nn.Sequential(*layers)


def calc_dimensions(input_size, kernel_size, stride, padding, dilation=1):
    return int((input_size + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1)

def calc_dimensions_final(in_dim, model_config: list):
    for layer in model_config:
        l_name = layer["layer"]
        if l_name == "Dropout" or l_name == "ReLU":
            continue
        elif l_name == "Conv2d" or l_name == "MaxPool2d":
            k = layer["params"].get("kernel_size", 1)
            p = layer["params"].get("padding", 0)
            s = layer["params"].get("stride", 1)
            in_dim = calc_dimensions(in_dim, k, s, p)
        elif l_name == "AdaptiveAvgPool2d":
            output_size = layer["params"]["output_size"]
            in_dim = output_size
        else:
            raise ValueError("Unsupported layer type in CNN config: {}".format(l_name))

    return in_dim

def get_final_out_channels(model_config: list):
    out_channels = 0
    for layer in model_config:
        l_name = layer["layer"]
        if l_name == "Conv2d":
            out_channels = layer["params"]["out_channels"]

    return out_channels


class CRNN(nn.Module):
    def __init__(self, img_h, img_w, in_dimensions_rnn=265, hidden_dimensions_rnn=128, num_classes=36):
        super().__init__()
        self.img_w_final = calc_dimensions_final(img_w, CNN_BASIC)
        img_h_final = calc_dimensions_final(img_h, CNN_BASIC)
        out_channels_cnn = get_final_out_channels(CNN_BASIC)

        self.cnn = parse_crrn_config(CNN_BASIC)
        self.flattening_layers = nn.ModuleList(
            [
                nn.Linear(out_channels_cnn * img_h_final, in_dimensions_rnn),
                nn.SiLU()
            ]
        )
        self.rnn = nn.LSTM(in_dimensions_rnn, hidden_dimensions_rnn, bidirectional=True)
        # hidden_dimensions_rnn * 2 because of bidirectional RNN ;  num_classes + 1 for the blank label in CTC loss
        self.fc_out = nn.Linear(hidden_dimensions_rnn * 2, num_classes + 1) # todo: check what that 1 is

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        for m in self.flattening_layers:
            x = m(x)
        # TODO: add Dropout here
        x, (_,_) = self.rnn(x)
        x = self.fc_out(x)

        return x

if __name__ == "__main__":
    # Example usage
    model = CRNN(img_h=50, img_w=200, in_dimensions_rnn=256, hidden_dimensions_rnn=128, num_classes=36)

    # Dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(8, 1, 50, 200)  # Batch size of 8
    output = model(dummy_input)
    assert  model.img_w_final == output.shape[0]
