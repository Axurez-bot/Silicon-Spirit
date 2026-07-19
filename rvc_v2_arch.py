"""
rvc_v2_arch.py — RVC v2 generator architecture (40 kHz, contentvec-based).

This is the SynthesizerTrn variant used by RVC v2 models trained in 2023
with contentvec features. The forward pass signature matches what
RVC-Project/Retrieval-based-Voice-Conversion-WebUI produces when training.

Reference: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# Building blocks
# ============================================================

class ResBlock1(torch.nn.Module):
    """Residual block with dilated convolutions. Used in the RVC upsampler."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=dilation[0] * (kernel_size - 1) // 2)
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=dilation[1] * (kernel_size - 1) // 2)
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=dilation[2] * (kernel_size - 1) // 2)
            ),
        ])
        self.convs1.apply(self._init_weights)

        self.convs2 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=1 * (kernel_size - 1) // 2)
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=1 * (kernel_size - 1) // 2)
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=1 * (kernel_size - 1) // 2)
            ),
        ])
        self.convs2.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    """Simpler residual block for the source part of the network."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=dilation[0] * (kernel_size - 1) // 2)
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=dilation[1] * (kernel_size - 1) // 2)
            ),
        ])
        self.convs.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x, x_mask):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt * x_mask)
            x = xt + x
        return x


class SourceModuleLayer(nn.Module):
    """Source module: combines harmonic + noise sources based on F0."""
    def __init__(self, hidden_channels, filter_channels, kernel_size, p_dropout, gin_channels, sr):
        super().__init__()
        self.sr = sr
        self.hidden_channels = hidden_channels

        # Harmonic source
        self.l_source = nn.Linear(hidden_channels, 1)

        # Noise source
        self.l_sin_gen = nn.Linear(hidden_channels, 1)
        self.l_linear = nn.Linear(hidden_channels, 1)

        # Filter for the noise
        self.filter = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(1, filter_channels, kernel_size, padding=kernel_size // 2)
        )
        self.filter.apply(self._init_weights)

        # Output mixing
        self.merge = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.merge.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x, x_mask, f0, uv, noise_scale=0.0):
        # x: [B, hidden, T], f0: [B, T], uv: [B, T] (unvoiced mask)
        x = x * x_mask
        f0 = f0.unsqueeze(1).float()
        uv = uv.unsqueeze(1).float()

        # Harmonic source: sum of sine waves at multiples of F0
        har = self.l_source(x).transpose(1, 2)
        har_source = self._sine_generate(har, f0, sr=self.sr)
        har_source = har_source * (1 - uv)
        har_source = har_source.transpose(1, 2)

        # Noise source
        noise = torch.randn_like(har_source) * noise_scale
        noi = self.l_sin_gen(x).transpose(1, 2) + self.l_linear(x).transpose(1, 2)
        noi_source = torch.sin(noi * f0) + noise
        noi_source = noi_source * uv
        noi_source = noi_source.transpose(1, 2)

        # Mix the sources
        x = har_source + noi_source
        x = self.filter(x) * x_mask
        x = self.merge(x) * x_mask
        return x

    @staticmethod
    def _sine_generate(har, f0, sr):
        """Generate harmonic stack from F0 using additive synthesis."""
        # har: [B, T, 1], f0: [B, 1, T]
        # Output: [B, 1, T]
        B, T, _ = har.shape
        # Generate 8 harmonics
        harmonics = []
        for n in range(1, 9):
            # phase accumulation
            phase = torch.cumsum(f0.transpose(1, 2) * n / sr, dim=-1) * 2 * math.pi
            sine = torch.sin(phase) * har
            harmonics.append(sine)
        return torch.sum(torch.stack(harmonics, dim=-1), dim=-1)  # [B, T, 1]


# ============================================================
# Main generator
# ============================================================

class GeneratorNSF(nn.Module):
    """RVC v2 generator with neural source filter (NSF) synthesis."""
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels, sr):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.sr = sr

        # Speaker embedding
        self.emb_g = nn.Embedding(109, gin_channels)
        nn.init.normal_(self.emb_g.weight, 0.0, 0.0)

        # Encoder: from spec_channels (HuBERT) to hidden
        self.enc_p = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(spec_channels, inter_channels, kernel_size=5, stride=1, padding=2)
        )
        self.enc_p.apply(self._init_weights)
        self.enc_p_f0 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(1, inter_channels, kernel_size=5, stride=1, padding=2)
        )
        self.enc_p_f0.apply(self._init_weights)

        # Downsampling
        self.down = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=2, padding=2, groups=inter_channels)
        )
        self.down.apply(self._init_weights)

        # Decoder: 4 ResBlocks
        self.dec = nn.ModuleList()
        for i, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.dec.append(ResBlock1(hidden_channels, k, d))
        self.dec.apply(self._init_weights)

        # Output projection
        self.out_proj = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.out_proj.apply(self._init_weights)

        # Source module
        self.source = SourceModuleLayer(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            kernel_size=7,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
            sr=sr,
        )

        # NSF (neural source filter) F0 embedding
        self.f0_emb = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        )
        self.f0_emb.apply(self._init_weights)

        # Upsample to audio rate (factor of 200 for 40kHz, 256 for 16kHz, etc.)
        self.upsample_rates = upsample_rates
        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r
        self.total_upsample = total_upsample
        self.ups = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        ch = hidden_channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.up_layers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(ch // 2, ch // 4, k, u, padding=(k - u) // 2)
                )
            )
            ch = ch // 2
        self.up_layers.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, z, f0, g=None):
        """
        z: [B, spec_channels, T_frames]  (HuBERT features, frame-rate)
        f0: [B, 1, T_audio]              (F0 at audio rate)
        g: [B, gin_channels, 1]          (speaker embedding, optional)
        """
        if g is None:
            g = self.emb_g.weight.unsqueeze(0).transpose(1, 2)  # [1, gin, 1]

        # Project speaker embedding and add to features
        g_expanded = g.expand(-1, -1, z.shape[-1])  # broadcast along time

        # Encode features
        z_p = self.enc_p(z) + self.enc_p_f0(self._interp_f0(f0, z.shape[-1]))
        z_p = z_p + g_expanded

        # Downsample
        z_p = self.down(z_p)

        # Generate harmonic+noise source
        x = torch.zeros_like(z_p)
        mask = torch.ones_like(z_p[:, :1, :])
        x = self.source(x, mask, f0.squeeze(1) if f0.shape[1] == 1 else f0, (f0 < 1.0).float().squeeze(1) if f0.shape[1] == 1 else (f0 < 1.0).float())

        # Add F0 embedding
        f0_emb = self.f0_emb(f0)
        f0_emb = F.interpolate(f0_emb, size=z_p.shape[-1], mode="linear")
        x = x + f0_emb

        # Decoder ResBlocks
        for block in self.dec:
            x = block(x)

        # Upsample to audio rate
        for up in self.up_layers:
            x = F.leaky_relu(x, 0.1)
            x = up(x)

        # Output
        x = F.leaky_relu(x, 0.1)
        x = self.out_proj(x)
        return torch.tanh(x)

    def _interp_f0(self, f0, target_len):
        """Resample F0 from audio rate to frame rate."""
        return F.interpolate(f0, size=target_len, mode="linear")


# ============================================================
# Loader
# ============================================================

def load_rvc_v2_generator(pth_path: str, device: str = "cuda"):
    """
    Load an RVC v2 model from a .pth file and return (generator, config_dict).
    The config_dict contains sr, f0_method, etc. for downstream use.
    """
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    cpt = ckpt.get("weight", ckpt)
    config = ckpt.get("config", [None])[0] if isinstance(ckpt.get("config"), list) else ckpt.get("config", {})

    sr = ckpt.get("sr", 40000)
    if not isinstance(sr, int):
        sr = 40000
    f0 = bool(ckpt.get("f0", True))
    version = ckpt.get("version", "v2")

    # Standard RVC v2 40kHz architecture params
    spec_channels = 768
    segment_size = 32
    inter_channels = 192
    hidden_channels = 192
    filter_channels = 768
    n_heads = 2
    n_layers = 6
    kernel_size = 3
    p_dropout = 0.0
    resblock_kernel_sizes = [3, 7]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]
    # Upsample from frame rate to 40kHz: 200x total
    # 8x * 5x * 5x = 200 (frame rate = 200 Hz)
    upsample_rates = [8, 5, 5]
    upsample_initial_channel = 32
    upsample_kernel_sizes = [16, 11, 11]
    gin_channels = 256

    if version == "v1":
        # v1 used 256x total upsample (frame rate 160 Hz) and different hidden
        upsample_rates = [4, 4, 4, 4]  # 256x
        upsample_kernel_sizes = [8, 8, 8, 8]
        hidden_channels = 256
        spec_channels = 768

    net = GeneratorNSF(
        spec_channels=spec_channels,
        segment_size=segment_size,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
        sr=sr,
    )

    # Load weights — RVC stores them under "weight" key
    sd = cpt if isinstance(cpt, dict) and "emb_g.weight" in cpt else cpt.get("model", cpt)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[RVC] Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[RVC] Unexpected keys (first 5): {unexpected[:5]}")

    net = net.to(device).eval()
    return net, {
        "sr": sr,
        "f0": f0,
        "version": version,
        "embedder_model": ckpt.get("embedder_model", "contentvec"),
        "model_name": ckpt.get("model_name", "unknown"),
    }
