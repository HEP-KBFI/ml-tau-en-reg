import torch
import torch.nn as nn




class PET(nn.Module):
    """Point Edge Transformer """
    def __init__(
        self,
        num_feat,
        num_jet,
        num_classes=2,
        num_keep=7,  # Number of features that wont be dropped
        feature_drop=0.1,
        projection_dim=128,
        local=True,
        K=10,
        num_local=2,
        num_layers=8,
        num_class_layers=2,
        num_gen_layers=2,
        num_heads=4,
        drop_probability=0.0,
        simple=False,
        layer_scale=True,
        layer_scale_init=1e-5,
        talking_head=False,
        mode='classifier',
        num_diffusion=3,
        dropout=0.0,
        class_activation=None,
     ):
        super().__init()
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init=layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.ema = 0.999
        self.class_activation = class_activation

        input_features = torch.zeros(1, num_feat)
        input_points = torch.zeros(1, 2)
        input_mask = torch.zeros(1, 1)
        input_jet = torch.zeros(num_jet)
        input_label = torch.zeros(num_classes)
        input_time = torch.zeros(1)

        # outputs_body = self.PET_body(
        #     input_features,
        #     input_points,
        #     input_mask,
        #     input_time,
        #     local=local,
        #     K=K,
        #     num_local=num_local,
        #     talking_head=talking_head
        # )

        self.body = 

        # outputs_classifier,outputs_regressor = self.PET_classifier(
        #     outputs_body,
        #     input_jet,
        #     num_class_layers=num_class_layers,
        #     num_jet=num_jet,
        #     simple=simple
        # )

        # outputs_generator = self.PET_generator(
        #     outputs_body,
        #     input_jet,
        #     label=input_label,
        #     time=input_time,
        #     mask=input_mask,
        #     num_layers=num_gen_layers,
        #     simple=simple,
        # )

        self.classifier_head = 

        self.generator_head = 

        self.classifier = 

        self.generator = 

        self.ema_generator_head = 

        self.ema_body = 


    # def PET_body(
    #     self,
    #     input_features,
    #     input_points,
    #     input_mask,
    #     input_time,
    #     local,
    #     K,
    #     num_local,
    #     talking_head,
    # ):
    #     # Randomly drop features not present in other datasets
    #     encoded  = RandomDrop(self.feature_drop if  'all' in self.mode else 0.0,num_skip=self.num_keep)(input_features)
    #     encoded = get_encoding(encoded,self.projection_dim)

    #     time = FourierProjection(input_time,self.projection_dim)
    #     time = tf.tile(time[:,None, :], [1,tf.shape(encoded)[1], 1])*input_mask
    #     time = layers.Dense(2*self.projection_dim,activation='gelu',use_bias=False)(time)
    #     scale,shift = tf.split(time,2,-1)
        
    #     encoded = encoded*(1.0+scale) + shift

        
    #     if local:
    #         coord_shift = tf.multiply(999., tf.cast(tf.equal(input_mask, 0), dtype='float32'))        
    #         points = input_points[:,:,:2]
    #         local_features = input_features
    #         for _ in range(num_local):
    #             local_features = get_neighbors(coord_shift+points,local_features,self.projection_dim,K)
    #             points = local_features
                
    #         encoded = layers.Add()([local_features,encoded])

    #     skip_connection = encoded
    #     for i in range(self.num_layers):
    #         x1 = layers.GroupNormalization(groups=1)(encoded)
    #         if talking_head:
    #             updates, _ = TalkingHeadAttention(self.projection_dim, self.num_heads, 0.0)(x1)
    #         else:
    #             updates = layers.MultiHeadAttention(num_heads=self.num_heads,
    #                                                 key_dim=self.projection_dim//self.num_heads)(x1,x1)

    #         if self.layer_scale:
    #             updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates,input_mask)
    #         updates = StochasticDepth(self.drop_probability)(updates)
            
    #         x2 = layers.Add()([updates,encoded])
    #         x3 = layers.GroupNormalization(groups=1)(x2)
    #         x3 = layers.Dense(2*self.projection_dim,activation="gelu")(x3)
    #         x3 = layers.Dropout(self.dropout)(x3)
    #         x3 = layers.Dense(self.projection_dim)(x3)
    #         if self.layer_scale:
    #             x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3,input_mask)
    #         x3 = StochasticDepth(self.drop_probability)(x3)
    #         encoded = layers.Add()([x3,x2])*input_mask
                    
    #     return encoded + skip_connection