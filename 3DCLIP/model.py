from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool3d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool3d(stride),
                nn.Conv3d(inplanes, planes * self.expansion, 1, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu3(out)
        return out

class ModifiedResNet3D(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=128, width=64):
        super().__init__()
        self.dropout = nn.Dropout3d(0.1)
        self.conv1 = nn.Conv3d(1, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool3D(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck3D(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck3D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class AttentionPool3D(nn.Module):
    def __init__(self, spatial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim ** 3 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # NCDHW -> (DHW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(0.1)),  
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(0.1))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 # text
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length + 1 #+1 from END token

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet3D(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None #self.build_attention_mask()
        )

        #self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, transformer_width),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(transformer_width, transformer_width),
            nn.Dropout(0.2)
        )
        self.eot_embedding = nn.Parameter(torch.randn(1, transformer_width))

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.eot_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        #initialize the feature embedding
        for layer in self.feature_embedding:
            if isinstance(layer, nn.Linear):
                # Following the CLIP MLP initialization convention
                std = layer.in_features ** -0.5
                nn.init.normal_(layer.weight, std=std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        
        if isinstance(self.visual, ModifiedResNet3D):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        '''
        context_features: tensor of shape [batch_size, 13]
        '''
        batch_size = text.size(0)

        # Embed each feature individually, sharing weights across features
        x = text.unsqueeze(-1)  # [batch_size, 13, 1]
        x = self.feature_embedding(x)  # [batch_size, 13, transformer_width]
        
        # Append EOT token embedding to the end
        eot = self.eot_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 1, transformer_width]
        x = torch.cat([x, eot], dim=1)  # [batch_size, 14, transformer_width]
        
        # Add positional embeddings
        x = x + self.positional_embedding  # [batch_size, 14, transformer_width]
        
        # Transformer expects [sequence_length, batch_size, transformer_width]
        x = x.permute(1, 0, 2)  # [14, batch_size, transformer_width]
        x = self.transformer(x)
        
        # Transform back to [batch_size, sequence_length, transformer_width]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        
        # Take the final token embedding (EOT token)
        final_token_embedding = x[:, -1, :]  # [batch_size, transformer_width]
        
        # Project into desired embedding space
        x = final_token_embedding @ self.text_projection  # [batch_size, output_dim]

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def extract_features(self, image, text):
        """
        Extracts image and text features without normalization or projection.
        Useful for intermediate feature extraction.
        """
        # extracting image features are easy, just pass through the visual encoder
        image_features = self.encode_image(image)

        #we extract both pooled and unpooled text features
        batch_size = text.size(0)

        # Embed each feature individually, sharing weights across features
        x = text.unsqueeze(-1)  # [batch_size, 13, 1]
        x = self.feature_embedding(x)  # [batch_size, 13, transformer_width]
        
        # Append EOT token embedding to the end
        eot = self.eot_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 1, transformer_width]
        x = torch.cat([x, eot], dim=1)  # [batch_size, 14, transformer_width]
        
        # Add positional embeddings
        x = x + self.positional_embedding  # [batch_size, 14, transformer_width]
        
        # Transformer expects [sequence_length, batch_size, transformer_width]
        x = x.permute(1, 0, 2)  # [14, batch_size, transformer_width]
        x = self.transformer(x)
        
        # Transform back to [batch_size, sequence_length, transformer_width]
        x = x.permute(1, 0, 2)
        unpooled_txt_features = self.ln_final(x) # keep in mind that [:,-1, :] is the EOT token embedding

        return image_features, unpooled_txt_features
    
    def get_clip_score(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        clip_cosine_score = (image_features * text_features).sum(dim=-1)  # shape [1], scalar

        return clip_cosine_score

    

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "eot_embedding", "positional_embedding"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 128                # Standard CLIP embedding dimension
    image_resolution = 128         # Your volume size (128³)
    vision_layers = (3, 4, 6, 3)   # Small ResNet50-like architecture (4 layers)
    vision_width = 64              # Standard width

    context_length = 13            # Number of context tokens (your shape descriptors)
    vocab_size = 100               # Artificial (for geometric descriptors, this is arbitrary)
    transformer_width = 256        # Width of transformer model for context encoder
    transformer_heads = 4          # Number of heads (256 dimension with 4 heads, 64 dim per head)
    transformer_layers = 4         # Moderate depth

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width,
        context_length, transformer_width, transformer_heads, transformer_layers
    ).to(device)
    convert_weights(model)
    model.eval()  # set model to evaluation mode

    # Generate random test tensors
    batch_size = 2  # Keep small for testing purposes
    dummy_volumes = torch.randn(batch_size, 1, 128, 128, 128).to(device)  # (N, C, D, H, W) in fp16

    # Generate dummy geometric context data as integers for simplicity
    dummy_context_tokens = torch.randn(batch_size, context_length).to(device).half()   # (N, context_length)
    with torch.no_grad():
        image_features = model.encode_image(dummy_volumes)
        context_features = model.encode_text(dummy_context_tokens)
        logits_per_image, logits_per_context = model(dummy_volumes, dummy_context_tokens)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    print("Image features shape:", image_features.shape)
    print("Context features shape:", context_features.shape)
    print("Logits per image shape:", logits_per_image.shape)
    print("Logits per context shape:", logits_per_context.shape)

