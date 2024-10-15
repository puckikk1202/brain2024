import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(patch_size * patch_size * in_channels, emb_dim)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 64, 1536)
        batch_size, channels, height, width = x.shape
        
        # Reshape to (batch_size, num_patches, patch_size*patch_size*channels)
        num_patches_w = width // self.patch_size
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
        
        # Apply linear projection
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, -1, self.patch_size * self.patch_size * channels)
        embeddings = self.linear_proj(patches)
        return embeddings

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout=0.5):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, emb_dim)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.layer_norm2(x)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, emb_dim, num_patches, num_output_tokens=257):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, num_output_tokens, emb_dim))
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.scale = emb_dim ** -0.5
    
    def forward(self, x):
        # x shape: (batch_size, num_patches, emb_dim)
        query = self.query.expand(x.size(0), -1, -1)
        key = self.key(x)
        value = self.value(x)
        
        attention = torch.bmm(query, key.transpose(1, 2)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(attention, value)  # Shape: (batch_size, num_output_tokens, emb_dim)
        return out


class ViT(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=1024, num_layers=6, num_heads=8, mlp_dim=512, num_classes=10, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (1536 // patch_size)*(64 // patch_size) + 1, emb_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.attn_pooling = AttentionPooling(emb_dim, (1536 // patch_size)*(64 // patch_size))

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(emb_dim),
        #     nn.Linear(emb_dim, num_classes)
        # )
    
    def forward(self, x):
        # x shape: (batch_size, 64, 1536), reshape to (batch_size, 1, 64, 1536)
        x = x.to(dtype=torch.float32)
        x = x.unsqueeze(1)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add cls token and positional embedding
        batch_size, num_patches, _ = x.shape
        # print('x', x.shape)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :num_patches + 1]
        x = self.dropout(x)
        
        # Transformer blocks
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, emb_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        out = self.attn_pooling(x)
        # print(x.shape)
        # Classification head
        # cls_output = x[0]  # Take the cls token output
        # print(x.shape)
        # out = self.mlp_head(cls_output)
        return out

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, time_len=224, patch_size=1, in_chans=128, embed_dim=256):
        super().__init__()
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x

class eeg_encoder(nn.Module):
    def __init__(self, time_len=1536, patch_size=4, embed_dim=1024, in_chans=64,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # print(x.shape)
        # print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 

class SeqTransformer(nn.Module):
    def __init__(self, ch=64, seq_len=96, emb_dim=512, num_layers=6, num_heads=8, mlp_dim=768, num_classes=8, dropout=0.5):
        super(SeqTransformer, self).__init__()
        self.embedding = nn.Linear(ch, emb_dim)  # Project channels to embedding dimension
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, ch=64, seq_len=1536)
        x = x.to(dtype=torch.float32)
        batch_size, ch, seq_len = x.shape
        
        # Linear projection of channels, reshape to (seq_len, batch_size, emb_dim)
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_len, ch)
        x = self.embedding(x)    # Apply embedding to (batch_size, seq_len, emb_dim)
        # x = x + self.pos_embedding  # Add position embedding
        x = self.dropout(x)
        
        # Transformer processing: (seq_len, batch_size, emb_dim)
        x = x.permute(1, 0, 2)
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Classification head: Take the output of the last token
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, emb_dim)
        cls_output = x.mean(dim=1)  # Pool over sequence dimension
        out = self.mlp_head(cls_output)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)

class EEGEncoder(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5, trainable=True):
        super(EEGEncoder, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224_in21k',
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            img_size=(64, 1536),
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.motion_projection = ProjectionHead(
            embedding_dim=768,
            projection_dim=256,
            dropout=0.5,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 8)
        )
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = self.motion_projection(x)
        x = self.mlp_head(x)
        return x

# # Example usage
if __name__=='__main__':
    batch_size = 16
    input_tensor = torch.randn(batch_size, 64, 1536)
    model = ViT(patch_size=32, emb_dim=768, num_classes=8)
    # model = EEGEncoder()    

    output = model(input_tensor)
    print(output.shape)  # Expected output shape: (batch_size, num_classes)
